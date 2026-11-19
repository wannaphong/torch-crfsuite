"""python-crfsuite compatible high-level API (Trainer, BaseTrainer, Tagger, ParsedDump)."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .crf1d import EncodedSequence, LinearChainCRF
from .features import FeatureItem, FeatureVocabulary, ItemSequence, ensure_item_sequence
from .trainers import ALGORITHM_ALIASES, TRAINER_DISPATCH, default_params_for


@dataclass
class ParsedDump:
    """CRFsuite model parameters returned by ``Tagger.info()``."""

    transitions: Dict[Tuple[str, str], float]
    state_features: Dict[Tuple[str, str], float]
    header: Dict[str, Any]
    labels: Dict[str, int]
    attributes: Dict[str, int]


class BaseTrainer:
    """Base trainer class matching ``pycrfsuite.BaseTrainer``."""

    def __init__(
        self,
        algorithm: str = "lbfgs",
        params: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.verbose = verbose
        self.device = torch.device(device)
        self.dtype = dtype
        self._dataset: List[Tuple[ItemSequence, List[str], int]] = []
        self._algorithm: str = "lbfgs"
        self._model_type: str = "crf1d"
        self._params: Dict[str, Any] = {}
        self.select(algorithm, "crf1d")
        if params:
            self.set_params(params)

    def select(self, algorithm: str, type: str = "crf1d") -> None:
        """Initialize the training algorithm and graphical model type."""
        if type.lower() != "crf1d":
            raise ValueError(f"Unsupported graphical model type: {type!r} (only 'crf1d')")
        key = ALGORITHM_ALIASES.get(algorithm.lower())
        if key is None:
            raise ValueError(
                f"Unknown training algorithm: {algorithm!r}. "
                f"Supported: {sorted(TRAINER_DISPATCH.keys())}"
            )
        self._algorithm = key
        self._model_type = "crf1d"
        self._params = default_params_for(key)

    def params(self) -> List[str]:
        """Return list of parameter names available for the current algorithm."""
        return list(self._params.keys())

    def set(self, name: str, value: Any) -> None:
        """Set a single training parameter."""
        if name not in self._params:
            raise KeyError(
                f"Unknown parameter {name!r} for algorithm {self._algorithm!r}. "
                f"Available: {self.params()}"
            )
        default_val = self._params[name]
        if isinstance(default_val, bool):
            if isinstance(value, str):
                self._params[name] = value.lower() in ("1", "true", "yes", "on")
            else:
                self._params[name] = bool(value)
        elif isinstance(default_val, int):
            self._params[name] = int(value)
        elif isinstance(default_val, float):
            self._params[name] = float(value)
        else:
            self._params[name] = str(value)

    def get(self, name: str) -> Any:
        """Get a single training parameter."""
        if name not in self._params:
            raise KeyError(f"Unknown parameter {name!r}")
        return self._params[name]

    def set_params(self, params: Dict[str, Any]) -> None:
        """Set multiple training parameters at once."""
        for k, v in params.items():
            self.set(k, v)

    def get_params(self) -> Dict[str, Any]:
        """Return a copy of the current training parameters."""
        return dict(self._params)

    def append(
        self,
        xseq: Sequence[FeatureItem] | ItemSequence,
        yseq: Sequence[str],
        group: int = 0,
    ) -> None:
        """Append an instance (item/label sequence) to the training dataset."""
        seq = ensure_item_sequence(xseq)
        labels = list(yseq)
        if len(seq) != len(labels):
            raise ValueError(
                f"Length mismatch: len(xseq)={len(seq)} != len(yseq)={len(labels)}"
            )
        self._dataset.append((seq, labels, int(group)))

    def clear(self) -> None:
        """Remove all appended training instances."""
        self._dataset.clear()

    def train(self, model: str | Path, holdout: int = -1) -> None:
        """Train the CRF model and save it to ``model`` (if non-empty)."""
        if not self._dataset:
            raise ValueError("Cannot train on an empty dataset; call append() first.")

        train_raw = [(x, y) for x, y, g in self._dataset if g != holdout]
        holdout_raw = [(x, y) for x, y, g in self._dataset if g == holdout]

        if not train_raw:
            raise ValueError(
                f"No training instances remain after excluding holdout group {holdout}."
            )

        vocab = FeatureVocabulary.from_dataset(
            train_raw,
            minfreq=float(self._params.get("feature.minfreq", 0.0)),
            possible_states=bool(self._params.get("feature.possible_states", False)),
            possible_transitions=bool(
                self._params.get("feature.possible_transitions", False)
            ),
        )

        train_seqs: List[EncodedSequence] = [
            vocab.encode_sequence(x, y, dtype=self.dtype).to(self.device)
            for x, y in train_raw
        ]
        holdout_seqs: List[EncodedSequence] = [
            vocab.encode_sequence(x, y, dtype=self.dtype).to(self.device)
            for x, y in holdout_raw
            if all(lbl in vocab.label_to_id for lbl in y)
        ]

        crf = LinearChainCRF(
            num_attributes=len(vocab.attr_to_id),
            num_labels=len(vocab.label_to_id),
            state_mask=vocab.state_mask,
            trans_mask=vocab.trans_mask,
            dtype=self.dtype,
        ).to(self.device)

        self._run_training(crf, vocab, train_seqs, holdout_seqs, model)

    def _run_training(
        self,
        crf: LinearChainCRF,
        vocab: FeatureVocabulary,
        train_seqs: Sequence[EncodedSequence],
        holdout_seqs: Sequence[EncodedSequence],
        model_path: str | Path,
    ) -> None:
        train_fn = TRAINER_DISPATCH[self._algorithm]
        train_fn(crf, train_seqs, holdout_seqs, self._params, on_iteration=None)
        if model_path:
            self._save_model(crf, vocab, model_path)

    def _save_model(
        self,
        crf: LinearChainCRF,
        vocab: FeatureVocabulary,
        model_path: str | Path,
    ) -> None:
        payload = {
            "format": "torch-crfsuite-v1",
            "algorithm": self._algorithm,
            "model_type": self._model_type,
            "params": self._params,
            "header": {
                "num_features": int(
                    (crf.effective_state_weights() != 0).sum().item()
                    + (crf.effective_trans_weights() != 0).sum().item()
                ),
                "num_labels": len(vocab.label_to_id),
                "num_attrs": len(vocab.attr_to_id),
                "algorithm": self._algorithm,
            },
            "attr_to_id": vocab.attr_to_id,
            "label_to_id": vocab.label_to_id,
            "state_mask": vocab.state_mask.cpu() if vocab.state_mask is not None else None,
            "trans_mask": vocab.trans_mask.cpu() if vocab.trans_mask is not None else None,
            "state_dict": {k: v.cpu() for k, v in crf.state_dict().items()},
        }
        torch.save(payload, str(model_path))


class Trainer(BaseTrainer):
    """Trainer with lifecycle hooks matching ``pycrfsuite.Trainer``."""

    def on_start(self, log: str = "") -> None:
        if self.verbose:
            print(f"[torch-crfsuite] Starting training ({self._algorithm})")

    def on_prepared(self, log: str = "") -> None:
        if self.verbose:
            print("[torch-crfsuite] Feature generation complete")

    def on_iteration(self, log: str, info: Dict[str, Any]) -> None:
        if self.verbose:
            msg = f"Iteration {info['num']}: loss={info['loss']:.6f}, time={info['time']:.3f}s"
            if "holdout" in info:
                h = info["holdout"]
                msg += f", holdout_acc={h['item_accuracy']:.4f}"
            print(msg)

    def on_optimization_end(self, log: str = "") -> None:
        if self.verbose:
            print("[torch-crfsuite] Optimization finished")

    def on_end(self, log: str = "") -> None:
        if self.verbose:
            print("[torch-crfsuite] Training complete")

    def _run_training(
        self,
        crf: LinearChainCRF,
        vocab: FeatureVocabulary,
        train_seqs: Sequence[EncodedSequence],
        holdout_seqs: Sequence[EncodedSequence],
        model_path: str | Path,
    ) -> None:
        self.on_start()
        self.on_prepared()
        train_fn = TRAINER_DISPATCH[self._algorithm]
        train_fn(
            crf,
            train_seqs,
            holdout_seqs,
            self._params,
            on_iteration=lambda info: self.on_iteration("", info),
        )
        self.on_optimization_end()
        if model_path:
            self._save_model(crf, vocab, model_path)
        self.on_end()


class Tagger:
    """Sequence tagger matching ``pycrfsuite.Tagger``."""

    def __init__(self, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self._crf: Optional[LinearChainCRF] = None
        self._vocab: Optional[FeatureVocabulary] = None
        self._header: Dict[str, Any] = {}
        self._current_seq: Optional[EncodedSequence] = None

    def open(self, name: str | Path) -> None:
        """Open a trained PyTorch CRFsuite model file."""
        payload = torch.load(str(name), map_location=self.device, weights_only=False)
        self._load_payload(payload)

    def open_inmemory(self, value: bytes) -> None:
        """Open a trained model from in-memory bytes."""
        buf = io.BytesIO(value)
        payload = torch.load(buf, map_location=self.device, weights_only=False)
        self._load_payload(payload)

    def _load_payload(self, payload: Dict[str, Any]) -> None:
        attr_to_id = payload["attr_to_id"]
        label_to_id = payload["label_to_id"]
        state_mask = payload.get("state_mask")
        trans_mask = payload.get("trans_mask")

        self._vocab = FeatureVocabulary(
            attr_to_id=attr_to_id,
            label_to_id=label_to_id,
            state_mask=state_mask,
            trans_mask=trans_mask,
        )
        self._header = dict(payload.get("header", {}))

        crf = LinearChainCRF(
            num_attributes=len(attr_to_id),
            num_labels=len(label_to_id),
            state_mask=state_mask,
            trans_mask=trans_mask,
        )
        crf.load_state_dict(payload["state_dict"])
        crf.to(self.device)
        crf.eval()
        self._crf = crf
        self._current_seq = None

    def close(self) -> None:
        """Release the loaded model."""
        self._crf = None
        self._vocab = None
        self._header = {}
        self._current_seq = None

    def __enter__(self) -> "Tagger":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _ensure_open(self) -> Tuple[LinearChainCRF, FeatureVocabulary]:
        if self._crf is None or self._vocab is None:
            raise RuntimeError("Tagger has no open model; call open() first.")
        return self._crf, self._vocab

    def labels(self) -> List[str]:
        """Return the list of labels known by the model."""
        _, vocab = self._ensure_open()
        return list(vocab.id_to_label)

    def set(self, xseq: Sequence[FeatureItem] | ItemSequence) -> None:
        """Set the current item sequence for tag(), probability(), and marginal()."""
        _, vocab = self._ensure_open()
        seq = ensure_item_sequence(xseq)
        self._current_seq = vocab.encode_sequence(seq).to(self.device)

    def tag(
        self, xseq: Optional[Sequence[FeatureItem] | ItemSequence] = None
    ) -> List[str]:
        """Predict the optimal label sequence via Viterbi decoding."""
        crf, vocab = self._ensure_open()
        if xseq is not None:
            self.set(xseq)
        if self._current_seq is None:
            raise RuntimeError("No sequence set; pass xseq or call set() first.")
        path_ids, _ = crf.viterbi_decode(self._current_seq)
        return [vocab.id_to_label[i] for i in path_ids]

    def probability(self, yseq: Sequence[str]) -> float:
        """Compute P(yseq | xseq) for the current item sequence."""
        crf, vocab = self._ensure_open()
        if self._current_seq is None:
            raise RuntimeError("No sequence set; call set() or tag() first.")
        if len(yseq) != self._current_seq.length:
            raise ValueError(
                f"Length mismatch: len(yseq)={len(yseq)} != {self._current_seq.length}"
            )
        y_ids = torch.tensor(
            [vocab.label_to_id[y] for y in yseq],
            dtype=torch.long,
            device=self.device,
        )
        with torch.no_grad():
            log_p = crf.log_probability(self._current_seq, y_ids)
        return float(torch.exp(log_p).item())

    def marginal(self, y: str, t: int) -> float:
        """Compute marginal probability P(y_t = y | xseq) at position t."""
        crf, vocab = self._ensure_open()
        if self._current_seq is None:
            raise RuntimeError("No sequence set; call set() or tag() first.")
        if y not in vocab.label_to_id:
            raise KeyError(f"Unknown label: {y!r}")
        if not (0 <= t < self._current_seq.length):
            raise IndexError(
                f"Position t={t} out of range for sequence length {self._current_seq.length}"
            )
        with torch.no_grad():
            marg = crf.marginals(self._current_seq)
        return float(marg[t, vocab.label_to_id[y]].item())

    def info(self) -> ParsedDump:
        """Return learned weights and metadata as a ``ParsedDump`` object."""
        crf, vocab = self._ensure_open()
        state_w = crf.effective_state_weights().detach().cpu()
        trans_w = crf.effective_trans_weights().detach().cpu()

        transitions: Dict[Tuple[str, str], float] = {}
        L = len(vocab.id_to_label)
        for i in range(L):
            for j in range(L):
                val = float(trans_w[i, j].item())
                if val != 0.0:
                    transitions[(vocab.id_to_label[i], vocab.id_to_label[j])] = val

        state_features: Dict[Tuple[str, str], float] = {}
        nz_rows, nz_cols = torch.nonzero(state_w, as_tuple=True)
        for r, c in zip(nz_rows.tolist(), nz_cols.tolist()):
            state_features[(vocab.id_to_attr[r], vocab.id_to_label[c])] = float(
                state_w[r, c].item()
            )

        return ParsedDump(
            transitions=transitions,
            state_features=state_features,
            header=dict(self._header),
            labels=dict(vocab.label_to_id),
            attributes=dict(vocab.attr_to_id),
        )

    def dump(self, file: Any) -> None:
        """Write a human-readable text dump of the model to ``file``."""
        parsed = self.info()
        file.write("HEADER = {\n")
        for k, v in parsed.header.items():
            file.write(f"  {k}: {v}\n")
        file.write("}\n\nLABELS = {\n")
        for k, v in parsed.labels.items():
            file.write(f"  {v}: {k}\n")
        file.write("}\n\nTRANSITIONS = {\n")
        for (src, dst), w in sorted(parsed.transitions.items()):
            file.write(f"  (1) {src} --> {dst}: {w:.6f}\n")
        file.write("}\n\nSTATE_FEATURES = {\n")
        for (attr, lbl), w in sorted(parsed.state_features.items()):
            file.write(f"  (0) {attr} --> {lbl}: {w:.6f}\n")
        file.write("}\n")

    def to_onnx(
        self,
        onnx_path: str | Path,
        opset_version: int = 14,
        save_sidecar: bool = True,
    ) -> Path:
        """Export the currently loaded CRF model to ONNX format."""
        from .onnx_export import export_to_onnx

        crf, vocab = self._ensure_open()
        return export_to_onnx(
            crf=crf,
            vocab=vocab,
            onnx_path=onnx_path,
            opset_version=opset_version,
            header=self._header,
            save_sidecar=save_sidecar,
        )
