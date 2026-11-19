"""ONNX export and ONNX Runtime inference for torch_crfsuite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .crf1d import LinearChainCRF
from .features import FeatureItem, FeatureVocabulary, ItemSequence, ensure_item_sequence


class ONNXCRFModule(nn.Module):
    """ONNX-exportable wrapper around a trained ``LinearChainCRF``.

    Takes a dense float32 feature matrix ``x`` of shape ``(T, num_attributes)``
    and computes:
      - ``best_path``: Int64 tensor of shape ``(T,)`` (Viterbi optimal label IDs)
      - ``marginals``: Float32 tensor of shape ``(T, num_labels)``
      - ``log_z``: Scalar float32 tensor (log partition function)
      - ``emissions``: Float32 tensor of shape ``(T, num_labels)``
    """

    def __init__(self, crf: LinearChainCRF) -> None:
        super().__init__()
        self.register_buffer(
            "state_weights",
            crf.effective_state_weights().detach().to(dtype=torch.float32).cpu(),
        )
        self.register_buffer(
            "trans_weights",
            crf.effective_trans_weights().detach().to(dtype=torch.float32).cpu(),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        T = x.shape[0]
        L = self.trans_weights.shape[0]
        emissions = torch.matmul(x, self.state_weights)

        # Viterbi forward pass
        v = emissions[0]
        bp_rows = torch.zeros((T, L), dtype=torch.long, device=x.device)
        for t in range(1, T):
            scores = v.unsqueeze(1) + self.trans_weights
            best_prev, bp_t = torch.max(scores, dim=0)
            v = best_prev + emissions[t]
            bp_rows[t] = bp_t

        # Viterbi backtrack
        best_path = torch.zeros((T,), dtype=torch.long, device=x.device)
        last_lbl = torch.argmax(v, dim=0)
        best_path[T - 1] = last_lbl
        for rev_i in range(1, T):
            t = T - rev_i
            last_lbl = bp_rows[t, last_lbl]
            best_path[t - 1] = last_lbl

        # Forward-Backward for marginals & log_z
        alphas = torch.zeros((T, L), dtype=emissions.dtype, device=x.device)
        alphas[0] = emissions[0]
        for t in range(1, T):
            prev = alphas[t - 1].unsqueeze(1) + self.trans_weights
            alphas[t] = torch.logsumexp(prev, dim=0) + emissions[t]
        log_z = torch.logsumexp(alphas[T - 1], dim=0)

        betas = torch.zeros((T, L), dtype=emissions.dtype, device=x.device)
        for rev_i in range(1, T):
            t = T - 1 - rev_i
            nxt = (emissions[t + 1] + betas[t + 1]).unsqueeze(0) + self.trans_weights
            betas[t] = torch.logsumexp(nxt, dim=1)

        marginals = torch.exp(alphas + betas - log_z)
        return best_path, marginals, log_z, emissions


def export_to_onnx(
    crf: LinearChainCRF,
    vocab: FeatureVocabulary,
    onnx_path: str | Path,
    opset_version: int = 14,
    header: Optional[Dict[str, Any]] = None,
    save_sidecar: bool = True,
) -> Path:
    """Export a trained ``LinearChainCRF`` and ``FeatureVocabulary`` to ONNX.

    Embeds vocabulary metadata inside ONNX ``metadata_props`` (when ``onnx`` is
    installed) and optionally writes a ``<onnx_path>.meta.json`` sidecar.
    """
    out_path = Path(onnx_path)
    wrapper = ONNXCRFModule(crf).eval()
    scripted = torch.jit.script(wrapper)

    num_attrs = max(1, len(vocab.attr_to_id))
    dummy_x = torch.zeros((3, num_attrs), dtype=torch.float32)

    torch.onnx.export(
        scripted,
        (dummy_x,),
        str(out_path),
        opset_version=opset_version,
        input_names=["x"],
        output_names=["best_path", "marginals", "log_z", "emissions"],
        dynamic_axes={
            "x": {0: "seq_len"},
            "best_path": {0: "seq_len"},
            "marginals": {0: "seq_len"},
            "emissions": {0: "seq_len"},
        },
    )

    meta = {
        "format": "torch-crfsuite-onnx-v1",
        "attr_to_id": vocab.attr_to_id,
        "label_to_id": vocab.label_to_id,
        "id_to_label": vocab.id_to_label,
        "trans_weights": crf.effective_trans_weights().detach().cpu().tolist(),
        "header": header or {},
    }

    try:
        import onnx

        model_proto = onnx.load(str(out_path))
        entry = model_proto.metadata_props.add()
        entry.key = "torch_crfsuite_meta"
        entry.value = json.dumps(meta)
        onnx.save(model_proto, str(out_path))
    except ImportError:
        pass

    if save_sidecar:
        meta_path = Path(f"{out_path}.meta.json")
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

    return out_path


class OnnxTagger:
    """Sequence tagger powered by ``onnxruntime`` matching ``Tagger`` API."""

    def __init__(self, providers: Optional[Sequence[str]] = None) -> None:
        self._providers = list(providers) if providers is not None else ["CPUExecutionProvider"]
        self._session: Any = None
        self._attr_to_id: Dict[str, int] = {}
        self._label_to_id: Dict[str, int] = {}
        self._id_to_label: List[str] = []
        self._trans_weights: Optional[np.ndarray] = None
        self._header: Dict[str, Any] = {}
        self._current_outputs: Optional[Tuple[np.ndarray, np.ndarray, float, np.ndarray]] = None

    def open(
        self,
        onnx_path: str | Path,
        meta_path: Optional[str | Path] = None,
    ) -> None:
        """Open an exported ONNX CRF model file using ``onnxruntime``."""
        import onnxruntime as ort

        path = Path(onnx_path)
        self._session = ort.InferenceSession(str(path), providers=self._providers)
        meta = self._read_metadata(self._session, path, meta_path)
        self._apply_metadata(meta)

    def open_inmemory(
        self,
        onnx_bytes: bytes,
        meta_json: Optional[str | bytes] = None,
    ) -> None:
        """Open an exported ONNX CRF model from in-memory bytes."""
        import onnxruntime as ort

        self._session = ort.InferenceSession(onnx_bytes, providers=self._providers)
        if meta_json is not None:
            raw = meta_json.decode("utf-8") if isinstance(meta_json, bytes) else meta_json
            meta = json.loads(raw)
        else:
            meta = self._read_metadata(self._session, None, None)
        self._apply_metadata(meta)

    def _read_metadata(
        self,
        session: Any,
        onnx_path: Optional[Path],
        meta_path: Optional[str | Path],
    ) -> Dict[str, Any]:
        custom_map = session.get_modelmeta().custom_metadata_map
        if "torch_crfsuite_meta" in custom_map:
            return json.loads(custom_map["torch_crfsuite_meta"])

        candidates: List[Path] = []
        if meta_path is not None:
            candidates.append(Path(meta_path))
        if onnx_path is not None:
            candidates.append(Path(f"{onnx_path}.meta.json"))
        for cand in candidates:
            if cand.exists():
                return json.loads(cand.read_text(encoding="utf-8"))

        raise RuntimeError(
            "Could not find torch_crfsuite metadata in ONNX model_meta or sidecar .meta.json."
        )

    def _apply_metadata(self, meta: Dict[str, Any]) -> None:
        self._attr_to_id = dict(meta["attr_to_id"])
        self._label_to_id = dict(meta["label_to_id"])
        self._id_to_label = list(meta["id_to_label"])
        self._trans_weights = np.asarray(meta["trans_weights"], dtype=np.float32)
        self._header = dict(meta.get("header", {}))
        self._current_outputs = None

    def close(self) -> None:
        self._session = None
        self._attr_to_id = {}
        self._label_to_id = {}
        self._id_to_label = []
        self._trans_weights = None
        self._header = {}
        self._current_outputs = None

    def __enter__(self) -> "OnnxTagger":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._session is None:
            raise RuntimeError("OnnxTagger has no open model; call open() first.")

    def labels(self) -> List[str]:
        self._ensure_open()
        return list(self._id_to_label)

    def _encode_dense(self, xseq: Sequence[FeatureItem] | ItemSequence) -> np.ndarray:
        seq = ensure_item_sequence(xseq)
        T = len(seq)
        A = max(1, len(self._attr_to_id))
        mat = np.zeros((T, A), dtype=np.float32)
        for t, attrs in enumerate(seq):
            for attr, val in attrs.items():
                aid = self._attr_to_id.get(attr)
                if aid is not None:
                    mat[t, aid] = float(val)
        return mat

    def set(self, xseq: Sequence[FeatureItem] | ItemSequence) -> None:
        """Run ONNX Runtime inference on ``xseq`` and cache outputs."""
        self._ensure_open()
        x_np = self._encode_dense(xseq)
        if x_np.shape[0] == 0:
            L = len(self._id_to_label)
            self._current_outputs = (
                np.zeros((0,), dtype=np.int64),
                np.zeros((0, L), dtype=np.float32),
                0.0,
                np.zeros((0, L), dtype=np.float32),
            )
            return
        best_path, marginals, log_z, emissions = self._session.run(
            ["best_path", "marginals", "log_z", "emissions"],
            {"x": x_np},
        )
        self._current_outputs = (
            best_path,
            marginals,
            float(np.asarray(log_z).item()),
            emissions,
        )

    def tag(
        self, xseq: Optional[Sequence[FeatureItem] | ItemSequence] = None
    ) -> List[str]:
        self._ensure_open()
        if xseq is not None:
            self.set(xseq)
        if self._current_outputs is None:
            raise RuntimeError("No sequence set; pass xseq or call set() first.")
        best_path = self._current_outputs[0]
        return [self._id_to_label[int(i)] for i in best_path.tolist()]

    def marginal(self, y: str, t: int) -> float:
        self._ensure_open()
        if self._current_outputs is None:
            raise RuntimeError("No sequence set; call set() or tag() first.")
        if y not in self._label_to_id:
            raise KeyError(f"Unknown label: {y!r}")
        marginals = self._current_outputs[1]
        if not (0 <= t < marginals.shape[0]):
            raise IndexError(
                f"Position t={t} out of range for sequence length {marginals.shape[0]}"
            )
        return float(marginals[t, self._label_to_id[y]])

    def probability(self, yseq: Sequence[str]) -> float:
        self._ensure_open()
        if self._current_outputs is None or self._trans_weights is None:
            raise RuntimeError("No sequence set; call set() or tag() first.")
        _, _, log_z, emissions = self._current_outputs
        T = emissions.shape[0]
        if len(yseq) != T:
            raise ValueError(f"Length mismatch: len(yseq)={len(yseq)} != {T}")
        if T == 0:
            return 1.0
        y_ids = [self._label_to_id[y] for y in yseq]
        score = sum(float(emissions[t, y_ids[t]]) for t in range(T))
        for t in range(1, T):
            score += float(self._trans_weights[y_ids[t - 1], y_ids[t]])
        return float(np.exp(score - log_z))
