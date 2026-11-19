"""Core 1st-order linear-chain Conditional Random Field (crf1d) in PyTorch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class EncodedSequence:
    """Sparse representation of an item sequence for PyTorch CRF computation.

    Attributes:
        length: Number of time steps T in the sequence.
        attr_indices: 2D LongTensor of shape (2, N) with (time_step, attr_id).
        attr_values: 1D FloatTensor of shape (N,) with attribute weights.
        label_ids: Optional 1D LongTensor of shape (T,) with gold label IDs.
    """

    length: int
    attr_indices: torch.Tensor
    attr_values: torch.Tensor
    label_ids: Optional[torch.Tensor] = None

    def to(self, device: torch.device | str) -> "EncodedSequence":
        return EncodedSequence(
            length=self.length,
            attr_indices=self.attr_indices.to(device),
            attr_values=self.attr_values.to(device),
            label_ids=self.label_ids.to(device) if self.label_ids is not None else None,
        )


class LinearChainCRF(nn.Module):
    """First-order linear-chain CRF (PyTorch equivalent of CRFsuite crf1d).

    Models the conditional probability:
        P(y | x) = exp(S(x, y)) / Z(x)
    where:
        S(x, y) = sum_t emit(x_t, y_t) + sum_{t=1}^{T-1} trans[y_{t-1}, y_t]

    Can be used standalone as an ``nn.Module`` in neural architectures or via
    the ``python-crfsuite``-compatible ``Trainer`` and ``Tagger`` interfaces.
    """

    def __init__(
        self,
        num_attributes: int,
        num_labels: int,
        state_mask: Optional[torch.Tensor] = None,
        trans_mask: Optional[torch.Tensor] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.num_attributes = num_attributes
        self.num_labels = num_labels

        self.state_weights = nn.Parameter(
            torch.zeros(num_attributes, num_labels, dtype=dtype)
        )
        self.trans_weights = nn.Parameter(
            torch.zeros(num_labels, num_labels, dtype=dtype)
        )

        if state_mask is not None:
            self.register_buffer("state_mask", state_mask.to(dtype=torch.bool))
        else:
            self.state_mask = None

        if trans_mask is not None:
            self.register_buffer("trans_mask", trans_mask.to(dtype=torch.bool))
        else:
            self.trans_mask = None

    def effective_state_weights(self) -> torch.Tensor:
        if self.state_mask is not None:
            return self.state_weights * self.state_mask.to(self.state_weights.dtype)
        return self.state_weights

    def effective_trans_weights(self) -> torch.Tensor:
        if self.trans_mask is not None:
            return self.trans_weights * self.trans_mask.to(self.trans_weights.dtype)
        return self.trans_weights

    def compute_emissions(self, seq: EncodedSequence) -> torch.Tensor:
        """Compute emission potential matrix of shape (T, L)."""
        T = seq.length
        L = self.num_labels
        w = self.effective_state_weights()
        if T == 0:
            return w.new_zeros((0, L))
        if seq.attr_indices.numel() == 0:
            return w.new_zeros((T, L))

        sparse_x = torch.sparse_coo_tensor(
            seq.attr_indices,
            seq.attr_values.to(dtype=w.dtype),
            size=(T, self.num_attributes),
            device=w.device,
        ).coalesce()
        return torch.sparse.mm(sparse_x, w)

    def sequence_score(
        self, emissions: torch.Tensor, label_ids: torch.Tensor
    ) -> torch.Tensor:
        """Compute unnormalized log-score S(x, y) for a label sequence."""
        T = emissions.shape[0]
        if T == 0:
            return emissions.new_tensor(0.0)
        t_idx = torch.arange(T, device=emissions.device)
        score = emissions[t_idx, label_ids].sum()
        if T > 1:
            trans = self.effective_trans_weights()
            score = score + trans[label_ids[:-1], label_ids[1:]].sum()
        return score

    def forward_alphas(
        self, emissions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the forward algorithm in log-space.

        Returns:
            alphas: Tensor of shape (T, L) with forward log-potentials.
            log_z: Scalar tensor with log partition function log Z(x).
        """
        T, L = emissions.shape
        if T == 0:
            return emissions.new_zeros((0, L)), emissions.new_tensor(0.0)

        trans = self.effective_trans_weights()
        alphas: List[torch.Tensor] = [emissions[0]]
        for t in range(1, T):
            # prev: (L, 1) + trans: (L, L) -> logsumexp over dim=0 -> (L,)
            prev = alphas[-1].unsqueeze(1) + trans
            alpha_t = torch.logsumexp(prev, dim=0) + emissions[t]
            alphas.append(alpha_t)

        alpha_mat = torch.stack(alphas, dim=0)
        log_z = torch.logsumexp(alpha_mat[-1], dim=0)
        return alpha_mat, log_z

    def backward_betas(self, emissions: torch.Tensor) -> torch.Tensor:
        """Run the backward algorithm in log-space.

        Returns:
            betas: Tensor of shape (T, L) with backward log-potentials.
        """
        T, L = emissions.shape
        if T == 0:
            return emissions.new_zeros((0, L))

        trans = self.effective_trans_weights()
        betas: List[torch.Tensor] = [emissions.new_zeros((L,))]
        for t in range(T - 2, -1, -1):
            # trans: (L, L) + (emissions[t+1] + betas[-1]): (1, L)
            nxt = (emissions[t + 1] + betas[-1]).unsqueeze(0) + trans
            beta_t = torch.logsumexp(nxt, dim=1)
            betas.append(beta_t)

        betas.reverse()
        return torch.stack(betas, dim=0)

    def neg_log_likelihood(self, seq: EncodedSequence) -> torch.Tensor:
        """Compute exact negative log-likelihood -log P(y | x) for one sequence."""
        if seq.label_ids is None:
            raise ValueError("EncodedSequence.label_ids is required for NLL.")
        emissions = self.compute_emissions(seq)
        _, log_z = self.forward_alphas(emissions)
        gold_score = self.sequence_score(emissions, seq.label_ids)
        return log_z - gold_score

    def log_probability(self, seq: EncodedSequence, label_ids: torch.Tensor) -> torch.Tensor:
        """Compute log P(y | x) = S(x, y) - log Z(x)."""
        emissions = self.compute_emissions(seq)
        _, log_z = self.forward_alphas(emissions)
        score = self.sequence_score(emissions, label_ids)
        return score - log_z

    def marginals(self, seq: EncodedSequence) -> torch.Tensor:
        """Compute marginal label probabilities P(y_t = l | x) of shape (T, L)."""
        emissions = self.compute_emissions(seq)
        if emissions.shape[0] == 0:
            return emissions
        alphas, log_z = self.forward_alphas(emissions)
        betas = self.backward_betas(emissions)
        log_marginals = alphas + betas - log_z
        return torch.exp(log_marginals)

    @torch.no_grad()
    def viterbi_decode(self, seq: EncodedSequence) -> Tuple[List[int], float]:
        """Find the highest-scoring label sequence via Viterbi decoding.

        Returns:
            (best_path, best_score)
        """
        T = seq.length
        if T == 0:
            return [], 0.0

        emissions = self.compute_emissions(seq)
        trans = self.effective_trans_weights()

        v = emissions[0]
        backpointers: List[torch.Tensor] = []

        for t in range(1, T):
            scores = v.unsqueeze(1) + trans
            best_prev_scores, bp_t = torch.max(scores, dim=0)
            v = best_prev_scores + emissions[t]
            backpointers.append(bp_t)

        best_score, last_label = torch.max(v, dim=0)
        best_path = [int(last_label.item())]
        for bp_t in reversed(backpointers):
            best_path.append(int(bp_t[best_path[-1]].item()))
        best_path.reverse()
        return best_path, float(best_score.item())
