"""PyTorch implementations of CRFsuite's 5 training algorithms.

Algorithms:
  - ``lbfgs``: Limited-memory BFGS (with optional L1/L2 Elastic Net regularization)
  - ``l2sgd``: Stochastic Gradient Descent with L2 regularization & LR calibration
  - ``ap``: Averaged Perceptron
  - ``pa``: Passive-Aggressive (PA, PA-I, PA-II)
  - ``arow``: Adaptive Regularization of Weight Vector (AROW)
"""

from __future__ import annotations

import math
import random
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from .crf1d import EncodedSequence, LinearChainCRF

ALGORITHMS = ("lbfgs", "l2sgd", "ap", "pa", "arow")

ALGORITHM_ALIASES = {
    "lbfgs": "lbfgs",
    "l-bfgs": "lbfgs",
    "l2sgd": "l2sgd",
    "sgd": "l2sgd",
    "ap": "ap",
    "averaged-perceptron": "ap",
    "pa": "pa",
    "passive-aggressive": "pa",
    "arow": "arow",
}


def default_params_for(algorithm: str) -> Dict[str, Any]:
    """Return default CRFsuite parameters for the given algorithm."""
    common: Dict[str, Any] = {
        "feature.minfreq": 0.0,
        "feature.possible_states": False,
        "feature.possible_transitions": False,
    }
    if algorithm == "lbfgs":
        return {
            **common,
            "c1": 0.0,
            "c2": 1.0,
            "max_iterations": 2147483647,
            "num_memories": 6,
            "epsilon": 1e-5,
            "period": 10,
            "delta": 1e-5,
            "linesearch": "StrongWolfe",
            "max_linesearch": 20,
        }
    if algorithm == "l2sgd":
        return {
            **common,
            "c2": 1.0,
            "max_iterations": 1000,
            "period": 10,
            "delta": 1e-5,
            "calibration.eta": 0.1,
            "calibration.rate": 2.0,
            "calibration.samples": 1000,
            "calibration.candidates": 10,
            "calibration.max_trials": 20,
        }
    if algorithm == "ap":
        return {
            **common,
            "max_iterations": 100,
            "epsilon": 1e-5,
        }
    if algorithm == "pa":
        return {
            **common,
            "type": 1,
            "c": 1.0,
            "error_sensitive": True,
            "averaging": True,
            "max_iterations": 100,
            "epsilon": 1e-5,
        }
    if algorithm == "arow":
        return {
            **common,
            "variance": 1.0,
            "gamma": 1.0,
            "max_iterations": 100,
            "epsilon": 1e-5,
        }
    raise ValueError(f"Unknown training algorithm: {algorithm!r}")


def evaluate_holdout(
    model: LinearChainCRF, holdout_seqs: Sequence[EncodedSequence]
) -> Dict[str, float]:
    """Evaluate label accuracy and instance accuracy on holdout sequences."""
    if not holdout_seqs:
        return {}
    correct_labels = 0
    total_labels = 0
    correct_seqs = 0
    total_loss = 0.0

    with torch.no_grad():
        for seq in holdout_seqs:
            if seq.label_ids is None or seq.length == 0:
                continue
            pred_path, _ = model.viterbi_decode(seq)
            gold = seq.label_ids.tolist()
            matches = sum(1 for p, g in zip(pred_path, gold) if p == g)
            correct_labels += matches
            total_labels += len(gold)
            if matches == len(gold):
                correct_seqs += 1
            total_loss += float(model.neg_log_likelihood(seq).item())

    return {
        "item_accuracy": correct_labels / max(1, total_labels),
        "instance_accuracy": correct_seqs / max(1, len(holdout_seqs)),
        "loss": total_loss,
    }


def _compute_feature_diff(
    model: LinearChainCRF,
    seq: EncodedSequence,
    gold_ids: torch.Tensor,
    pred_ids: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute sparse/dense feature difference phi(x, y_gold) - phi(x, y_pred)."""
    device = model.state_weights.device
    dtype = model.state_weights.dtype
    pred_tensor = torch.tensor(pred_ids, dtype=torch.long, device=device)

    d_state = torch.zeros_like(model.state_weights)
    d_trans = torch.zeros_like(model.trans_weights)

    if seq.attr_indices.numel() > 0:
        t_idx = seq.attr_indices[0]
        a_idx = seq.attr_indices[1]
        vals = seq.attr_values.to(dtype=dtype)

        g_labels = gold_ids[t_idx]
        p_labels = pred_tensor[t_idx]
        d_state.index_put_((a_idx, g_labels), vals, accumulate=True)
        d_state.index_put_((a_idx, p_labels), -vals, accumulate=True)

    if seq.length > 1:
        ones = torch.ones(seq.length - 1, dtype=dtype, device=device)
        d_trans.index_put_((gold_ids[:-1], gold_ids[1:]), ones, accumulate=True)
        d_trans.index_put_((pred_tensor[:-1], pred_tensor[1:]), -ones, accumulate=True)

    if model.state_mask is not None:
        d_state = d_state * model.state_mask.to(dtype)
    if model.trans_mask is not None:
        d_trans = d_trans * model.trans_mask.to(dtype)

    return d_state, d_trans


def _apply_proximal_l1(model: LinearChainCRF, lr_c1: float) -> None:
    """Soft-thresholding proximal step for L1 regularization."""
    if lr_c1 <= 0.0:
        return
    with torch.no_grad():
        for param in (model.state_weights, model.trans_weights):
            param.copy_(torch.sign(param) * torch.clamp(param.abs() - lr_c1, min=0.0))


def train_lbfgs(
    model: LinearChainCRF,
    train_seqs: Sequence[EncodedSequence],
    holdout_seqs: Sequence[EncodedSequence],
    params: Dict[str, Any],
    on_iteration: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> None:
    """Train LinearChainCRF using L-BFGS."""
    c1 = float(params.get("c1", 0.0))
    c2 = float(params.get("c2", 1.0))
    max_iters = min(int(params.get("max_iterations", 1000)), 10000)
    num_memories = int(params.get("num_memories", 6))
    epsilon = float(params.get("epsilon", 1e-5))
    period = int(params.get("period", 10))
    delta = float(params.get("delta", 1e-5))
    max_ls = int(params.get("max_linesearch", 20))
    ls_name = str(params.get("linesearch", "StrongWolfe"))
    line_search_fn = "strong_wolfe" if "wolfe" in ls_name.lower() else None

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=1,
        max_eval=max_ls,
        history_size=num_memories,
        line_search_fn=line_search_fn,
    )

    loss_history: List[float] = []

    for it in range(1, max_iters + 1):
        t0 = time.perf_counter()

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            nll = model.state_weights.new_tensor(0.0)
            for seq in train_seqs:
                nll = nll + model.neg_log_likelihood(seq)
            if c2 > 0.0:
                reg_l2 = 0.5 * c2 * (
                    model.effective_state_weights().pow(2).sum()
                    + model.effective_trans_weights().pow(2).sum()
                )
                nll = nll + reg_l2
            if c1 > 0.0:
                reg_l1 = c1 * (
                    model.effective_state_weights().abs().sum()
                    + model.effective_trans_weights().abs().sum()
                )
                nll = nll + reg_l1
            nll.backward()
            return nll

        loss_tensor = optimizer.step(closure)
        if c1 > 0.0:
            _apply_proximal_l1(model, c1 * 1e-3)

        loss_val = float(loss_tensor.item())
        loss_history.append(loss_val)

        grad_norm = math.sqrt(
            sum(
                float(p.grad.pow(2).sum().item())
                for p in model.parameters()
                if p.grad is not None
            )
        )
        param_norm = math.sqrt(
            sum(float(p.detach().pow(2).sum().item()) for p in model.parameters())
        )

        info: Dict[str, Any] = {
            "num": it,
            "loss": loss_val,
            "feature_norm": param_norm,
            "grad_norm": grad_norm,
            "time": time.perf_counter() - t0,
        }
        if holdout_seqs:
            info["holdout"] = evaluate_holdout(model, holdout_seqs)
        if on_iteration is not None:
            on_iteration(info)

        if grad_norm / max(1.0, param_norm) <= epsilon:
            break
        if len(loss_history) > period:
            prev = loss_history[-period - 1]
            if abs(prev - loss_val) / max(1.0, abs(prev)) < delta:
                break


def _calibrate_l2sgd_eta(
    model: LinearChainCRF,
    train_seqs: Sequence[EncodedSequence],
    c2: float,
    params: Dict[str, Any],
) -> float:
    """Calibrate initial learning rate eta0 using CRFsuite's calibration heuristic."""
    eta = float(params.get("calibration.eta", 0.1))
    rate = float(params.get("calibration.rate", 2.0))
    n_samples = min(len(train_seqs), int(params.get("calibration.samples", 1000)))
    n_candidates = int(params.get("calibration.candidates", 10))
    max_trials = int(params.get("calibration.max_trials", 20))

    if n_samples == 0:
        return eta

    sample_seqs = random.sample(list(train_seqs), n_samples)
    lam = c2 / max(1, len(train_seqs))

    saved_state = model.state_weights.detach().clone()
    saved_trans = model.trans_weights.detach().clone()

    def eval_sample_loss() -> float:
        with torch.no_grad():
            total = 0.0
            for seq in sample_seqs:
                total += float(model.neg_log_likelihood(seq).item())
            total += 0.5 * lam * len(sample_seqs) * float(
                model.effective_state_weights().pow(2).sum().item()
                + model.effective_trans_weights().pow(2).sum().item()
            )
            return total

    initial_loss = eval_sample_loss()
    best_eta = eta
    best_loss = initial_loss
    candidates_found = 0

    for _ in range(max_trials):
        with torch.no_grad():
            model.state_weights.copy_(saved_state)
            model.trans_weights.copy_(saved_trans)

        for seq in sample_seqs:
            model.zero_grad()
            loss = model.neg_log_likelihood(seq)
            if lam > 0.0:
                loss = loss + 0.5 * lam * (
                    model.effective_state_weights().pow(2).sum()
                    + model.effective_trans_weights().pow(2).sum()
                )
            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p.sub_(eta * p.grad)

        cur_loss = eval_sample_loss()
        if cur_loss < initial_loss:
            candidates_found += 1
            if cur_loss < best_loss:
                best_loss = cur_loss
                best_eta = eta
            eta *= rate
        else:
            eta /= rate

        if candidates_found >= n_candidates:
            break

    with torch.no_grad():
        model.state_weights.copy_(saved_state)
        model.trans_weights.copy_(saved_trans)

    return best_eta


def train_l2sgd(
    model: LinearChainCRF,
    train_seqs: Sequence[EncodedSequence],
    holdout_seqs: Sequence[EncodedSequence],
    params: Dict[str, Any],
    on_iteration: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> None:
    """Train LinearChainCRF using L2-regularized SGD."""
    c2 = float(params.get("c2", 1.0))
    max_iters = int(params.get("max_iterations", 1000))
    period = int(params.get("period", 10))
    delta = float(params.get("delta", 1e-5))

    n = max(1, len(train_seqs))
    lam = c2 / n
    eta0 = _calibrate_l2sgd_eta(model, train_seqs, c2, params)
    t_step = 0
    loss_history: List[float] = []
    indices = list(range(len(train_seqs)))

    for epoch in range(1, max_iters + 1):
        t0 = time.perf_counter()
        random.shuffle(indices)
        epoch_loss = 0.0

        for idx in indices:
            seq = train_seqs[idx]
            eta = eta0 / (1.0 + lam * eta0 * t_step)
            t_step += 1

            model.zero_grad()
            nll = model.neg_log_likelihood(seq)
            reg = 0.5 * lam * (
                model.effective_state_weights().pow(2).sum()
                + model.effective_trans_weights().pow(2).sum()
            )
            loss = nll + reg
            loss.backward()
            epoch_loss += float(loss.item())

            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p.sub_(eta * p.grad)

        loss_history.append(epoch_loss)
        info: Dict[str, Any] = {
            "num": epoch,
            "loss": epoch_loss,
            "eta": eta0 / (1.0 + lam * eta0 * max(1, t_step)),
            "time": time.perf_counter() - t0,
        }
        if holdout_seqs:
            info["holdout"] = evaluate_holdout(model, holdout_seqs)
        if on_iteration is not None:
            on_iteration(info)

        if len(loss_history) > period:
            prev = loss_history[-period - 1]
            if abs(prev - epoch_loss) / max(1.0, abs(prev)) < delta:
                break


def train_ap(
    model: LinearChainCRF,
    train_seqs: Sequence[EncodedSequence],
    holdout_seqs: Sequence[EncodedSequence],
    params: Dict[str, Any],
    on_iteration: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> None:
    """Train LinearChainCRF using Averaged Perceptron."""
    max_iters = int(params.get("max_iterations", 100))
    epsilon = float(params.get("epsilon", 1e-5))

    acc_state = torch.zeros_like(model.state_weights)
    acc_trans = torch.zeros_like(model.trans_weights)
    c_updates = 1

    indices = list(range(len(train_seqs)))
    for epoch in range(1, max_iters + 1):
        t0 = time.perf_counter()
        random.shuffle(indices)
        num_errors = 0
        total_items = 0

        for idx in indices:
            seq = train_seqs[idx]
            if seq.label_ids is None or seq.length == 0:
                continue
            pred_ids, _ = model.viterbi_decode(seq)
            gold_list = seq.label_ids.tolist()
            total_items += seq.length
            err = sum(1 for p, g in zip(pred_ids, gold_list) if p != g)

            if err > 0:
                num_errors += err
                d_state, d_trans = _compute_feature_diff(
                    model, seq, seq.label_ids, pred_ids
                )
                with torch.no_grad():
                    model.state_weights.add_(d_state)
                    model.trans_weights.add_(d_trans)
                    acc_state.add_(d_state, alpha=float(c_updates))
                    acc_trans.add_(d_trans, alpha=float(c_updates))
            c_updates += 1

        error_rate = num_errors / max(1, total_items)
        info: Dict[str, Any] = {
            "num": epoch,
            "loss": float(num_errors),
            "error_rate": error_rate,
            "time": time.perf_counter() - t0,
        }
        if holdout_seqs:
            info["holdout"] = evaluate_holdout(model, holdout_seqs)
        if on_iteration is not None:
            on_iteration(info)

        if error_rate <= epsilon:
            break

    with torch.no_grad():
        model.state_weights.sub_(acc_state / float(max(1, c_updates)))
        model.trans_weights.sub_(acc_trans / float(max(1, c_updates)))


def train_pa(
    model: LinearChainCRF,
    train_seqs: Sequence[EncodedSequence],
    holdout_seqs: Sequence[EncodedSequence],
    params: Dict[str, Any],
    on_iteration: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> None:
    """Train LinearChainCRF using Passive-Aggressive (PA, PA-I, PA-II)."""
    pa_type = int(params.get("type", 1))
    c_aggr = float(params.get("c", 1.0))
    error_sensitive = bool(params.get("error_sensitive", True))
    averaging = bool(params.get("averaging", True))
    max_iters = int(params.get("max_iterations", 100))
    epsilon = float(params.get("epsilon", 1e-5))

    acc_state = torch.zeros_like(model.state_weights)
    acc_trans = torch.zeros_like(model.trans_weights)
    c_updates = 1

    indices = list(range(len(train_seqs)))
    for epoch in range(1, max_iters + 1):
        t0 = time.perf_counter()
        random.shuffle(indices)
        sum_loss = 0.0
        num_items = 0

        for idx in indices:
            seq = train_seqs[idx]
            if seq.label_ids is None or seq.length == 0:
                continue
            num_items += seq.length
            pred_ids, pred_score = model.viterbi_decode(seq)
            gold_list = seq.label_ids.tolist()
            err = sum(1 for p, g in zip(pred_ids, gold_list) if p != g)
            if err == 0:
                continue

            with torch.no_grad():
                emissions = model.compute_emissions(seq)
                gold_score = float(model.sequence_score(emissions, seq.label_ids).item())

            cost = math.sqrt(float(err)) if error_sensitive else 1.0
            loss = (pred_score - gold_score) + cost
            if loss <= 0.0:
                continue

            sum_loss += loss
            d_state, d_trans = _compute_feature_diff(
                model, seq, seq.label_ids, pred_ids
            )
            norm_sq = float(d_state.pow(2).sum().item() + d_trans.pow(2).sum().item())
            if norm_sq <= 0.0:
                continue

            if pa_type == 0:
                tau = loss / norm_sq
            elif pa_type == 1:
                tau = min(c_aggr, loss / norm_sq)
            elif pa_type == 2:
                tau = loss / (norm_sq + 0.5 / c_aggr)
            else:
                raise ValueError(f"Unsupported PA type: {pa_type!r} (expected 0, 1, or 2)")

            with torch.no_grad():
                model.state_weights.add_(d_state, alpha=tau)
                model.trans_weights.add_(d_trans, alpha=tau)
                if averaging:
                    acc_state.add_(d_state, alpha=tau * float(c_updates))
                    acc_trans.add_(d_trans, alpha=tau * float(c_updates))
            c_updates += 1

        mean_loss = sum_loss / max(1, len(train_seqs))
        info: Dict[str, Any] = {
            "num": epoch,
            "loss": sum_loss,
            "avg_loss": mean_loss,
            "time": time.perf_counter() - t0,
        }
        if holdout_seqs:
            info["holdout"] = evaluate_holdout(model, holdout_seqs)
        if on_iteration is not None:
            on_iteration(info)

        if mean_loss <= epsilon:
            break

    if averaging:
        with torch.no_grad():
            model.state_weights.sub_(acc_state / float(max(1, c_updates)))
            model.trans_weights.sub_(acc_trans / float(max(1, c_updates)))


def train_arow(
    model: LinearChainCRF,
    train_seqs: Sequence[EncodedSequence],
    holdout_seqs: Sequence[EncodedSequence],
    params: Dict[str, Any],
    on_iteration: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> None:
    """Train LinearChainCRF using Adaptive Regularization of Weight Vector (AROW)."""
    variance = float(params.get("variance", 1.0))
    gamma = float(params.get("gamma", 1.0))
    max_iters = int(params.get("max_iterations", 100))
    epsilon = float(params.get("epsilon", 1e-5))

    sigma_state = torch.full_like(model.state_weights, variance)
    sigma_trans = torch.full_like(model.trans_weights, variance)

    indices = list(range(len(train_seqs)))
    for epoch in range(1, max_iters + 1):
        t0 = time.perf_counter()
        random.shuffle(indices)
        sum_loss = 0.0

        for idx in indices:
            seq = train_seqs[idx]
            if seq.label_ids is None or seq.length == 0:
                continue
            pred_ids, pred_score = model.viterbi_decode(seq)
            gold_list = seq.label_ids.tolist()
            err = sum(1 for p, g in zip(pred_ids, gold_list) if p != g)
            if err == 0:
                continue

            with torch.no_grad():
                emissions = model.compute_emissions(seq)
                gold_score = float(model.sequence_score(emissions, seq.label_ids).item())

            margin = gold_score - pred_score
            cost = math.sqrt(float(err))
            loss = cost - margin
            if loss <= 0.0:
                continue

            sum_loss += loss
            d_state, d_trans = _compute_feature_diff(
                model, seq, seq.label_ids, pred_ids
            )

            # Diagonal covariance confidence v = d^T Sigma d
            v = float(
                (sigma_state * d_state.pow(2)).sum().item()
                + (sigma_trans * d_trans.pow(2)).sum().item()
            )
            beta = 1.0 / (v + gamma)
            alpha = max(0.0, loss) * beta

            with torch.no_grad():
                model.state_weights.add_(sigma_state * d_state, alpha=alpha)
                model.trans_weights.add_(sigma_trans * d_trans, alpha=alpha)
                # Sigma <- Sigma - beta * (Sigma d)^2
                sigma_state.sub_(beta * (sigma_state * d_state).pow(2))
                sigma_trans.sub_(beta * (sigma_trans * d_trans).pow(2))
                sigma_state.clamp_(min=1e-8)
                sigma_trans.clamp_(min=1e-8)

        mean_loss = sum_loss / max(1, len(train_seqs))
        info: Dict[str, Any] = {
            "num": epoch,
            "loss": sum_loss,
            "avg_loss": mean_loss,
            "time": time.perf_counter() - t0,
        }
        if holdout_seqs:
            info["holdout"] = evaluate_holdout(model, holdout_seqs)
        if on_iteration is not None:
            on_iteration(info)

        if mean_loss <= epsilon:
            break


TRAINER_DISPATCH = {
    "lbfgs": train_lbfgs,
    "l2sgd": train_l2sgd,
    "ap": train_ap,
    "pa": train_pa,
    "arow": train_arow,
}
