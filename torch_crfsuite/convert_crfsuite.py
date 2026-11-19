"""Convert native CRFsuite binary (.model) or dump (.txt) files to ONNX."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from .crf1d import LinearChainCRF
from .features import FeatureVocabulary
from .onnx_export import export_to_onnx

_SECTION_RE = re.compile(r"^([A-Z0-9_]+)\s*=\s*\{\s*$")
_KV_RE = re.compile(r"^\s*([^:]+?):\s*(.+?)\s*$")
_TRANS_RE = re.compile(r"^\s*\(\d+\)\s+(.+?)\s+-->\s+(.+?):\s+([-+0-9.eE]+)\s*$")
_STATE_RE = re.compile(r"^\s*\(\d+\)\s+(.*?)\s+-->\s+(.+?):\s+([-+0-9.eE]+)\s*$")


def parse_crfsuite_dump(text: str) -> Dict[str, Any]:
    """Parse text output from ``crfsuite dump <model>`` or ``Tagger.dump()``."""
    header: Dict[str, Any] = {}
    labels: Dict[str, int] = {}
    attributes: Dict[str, int] = {}
    transitions: Dict[Tuple[str, str], float] = {}
    state_features: Dict[Tuple[str, str], float] = {}

    section: Optional[str] = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip("\r\n")
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == "}":
            section = None
            continue

        sec_match = _SECTION_RE.match(stripped)
        if sec_match:
            section = sec_match.group(1)
            continue

        if section in ("FILEHEADER", "HEADER"):
            m = _KV_RE.match(line)
            if m:
                header[m.group(1).strip()] = m.group(2).strip()
        elif section == "LABELS":
            m = _KV_RE.match(line)
            if m:
                idx = int(m.group(1).strip())
                lbl = m.group(2).strip()
                labels[lbl] = idx
        elif section == "ATTRIBUTES":
            m = _KV_RE.match(line)
            if m:
                idx = int(m.group(1).strip())
                attr = m.group(2).strip()
                attributes[attr] = idx
        elif section == "TRANSITIONS":
            m = _TRANS_RE.match(line)
            if m:
                transitions[(m.group(1), m.group(2))] = float(m.group(3))
        elif section == "STATE_FEATURES":
            m = _STATE_RE.match(line)
            if m:
                state_features[(m.group(1), m.group(2))] = float(m.group(3))

    if not labels:
        raise ValueError("No LABELS section found in CRFsuite dump.")

    # Re-index labels and attributes into contiguous 0..N-1 IDs
    sorted_labels = [k for k, _ in sorted(labels.items(), key=lambda kv: kv[1])]
    label_to_id = {lbl: i for i, lbl in enumerate(sorted_labels)}

    if attributes:
        sorted_attrs = [k for k, _ in sorted(attributes.items(), key=lambda kv: kv[1])]
    else:
        sorted_attrs = sorted({attr for attr, _ in state_features.keys()})
    attr_to_id = {attr: i for i, attr in enumerate(sorted_attrs)}

    return {
        "header": header,
        "label_to_id": label_to_id,
        "attr_to_id": attr_to_id,
        "transitions": transitions,
        "state_features": state_features,
    }


def _load_via_pycrfsuite(model_path: Path) -> Dict[str, Any]:
    import pycrfsuite  # type: ignore

    tagger = pycrfsuite.Tagger()
    tagger.open(str(model_path))
    try:
        info = tagger.info()
        sorted_labels = [k for k, _ in sorted(info.labels.items(), key=lambda kv: kv[1])]
        sorted_attrs = [k for k, _ in sorted(info.attributes.items(), key=lambda kv: kv[1])]
        return {
            "header": dict(info.header),
            "label_to_id": {lbl: i for i, lbl in enumerate(sorted_labels)},
            "attr_to_id": {attr: i for i, attr in enumerate(sorted_attrs)},
            "transitions": dict(info.transitions),
            "state_features": dict(info.state_features),
        }
    finally:
        tagger.close()


def load_crfsuite_model(input_path: str | Path) -> Dict[str, Any]:
    """Load a CRFsuite binary ``.model`` or text ``dump`` file.

    Resolution order:
      1. If ``input_path`` is a text dump (e.g. ``.txt`` / ``.dump`` or starts
         with ``FILEHEADER`` / ``HEADER``), parse it directly in pure Python.
      2. Otherwise, try ``pycrfsuite.Tagger().open(...)``.
      3. Fall back to ``crfsuite dump <input_path>`` if the ``crfsuite`` CLI is
         available on ``PATH``.
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"CRFsuite model file not found: {path}")

    with path.open("rb") as f:
        magic = f.read(16)

    is_text_dump = (
        path.suffix.lower() in (".txt", ".dump")
        or magic.startswith(b"FILEHEADER")
        or magic.startswith(b"HEADER")
    )
    if is_text_dump:
        return parse_crfsuite_dump(path.read_text(encoding="utf-8"))

    try:
        return _load_via_pycrfsuite(path)
    except ImportError:
        pass

    crfsuite_bin = shutil.which("crfsuite")
    if crfsuite_bin is not None:
        proc = subprocess.run(
            [crfsuite_bin, "dump", str(path)],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        return parse_crfsuite_dump(proc.stdout)

    raise RuntimeError(
        f"Cannot read binary CRFsuite model {path!s}: neither 'python-crfsuite' "
        "nor the 'crfsuite' CLI was found. Install 'python-crfsuite' "
        "(pip install python-crfsuite) or convert via 'crfsuite dump model.model > model.dump'."
    )


def convert_crfsuite_to_onnx(
    input_path: str | Path,
    onnx_path: str | Path,
    opset_version: int = 14,
    save_sidecar: bool = True,
) -> Path:
    """Convert a CRFsuite ``.model`` (or dump file) to an ONNX model."""
    parsed = load_crfsuite_model(input_path)
    attr_to_id: Dict[str, int] = parsed["attr_to_id"]
    label_to_id: Dict[str, int] = parsed["label_to_id"]

    vocab = FeatureVocabulary(attr_to_id=attr_to_id, label_to_id=label_to_id)
    crf = LinearChainCRF(
        num_attributes=len(attr_to_id),
        num_labels=len(label_to_id),
        dtype=torch.float32,
    )

    with torch.no_grad():
        for (attr, lbl), weight in parsed["state_features"].items():
            aid = attr_to_id.get(attr)
            lid = label_to_id.get(lbl)
            if aid is not None and lid is not None:
                crf.state_weights[aid, lid] = float(weight)

        for (src_lbl, dst_lbl), weight in parsed["transitions"].items():
            sid = label_to_id.get(src_lbl)
            did = label_to_id.get(dst_lbl)
            if sid is not None and did is not None:
                crf.trans_weights[sid, did] = float(weight)

    return export_to_onnx(
        crf=crf,
        vocab=vocab,
        onnx_path=onnx_path,
        opset_version=opset_version,
        header=parsed["header"],
        save_sidecar=save_sidecar,
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="crfsuite2onnx",
        description="Convert a CRFsuite .model (e.g. crf-wordseg.model) or dump file to ONNX.",
    )
    parser.add_argument("input", help="Path to CRFsuite .model or crfsuite dump .txt/.dump file")
    parser.add_argument("output", help="Destination .onnx file path")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version (default: 14)")
    parser.add_argument(
        "--no-sidecar",
        action="store_true",
        help="Do not write <output>.meta.json sidecar file",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify exported ONNX model loads and runs in onnxruntime",
    )
    args = parser.parse_args(argv)

    out_path = convert_crfsuite_to_onnx(
        input_path=args.input,
        onnx_path=args.output,
        opset_version=args.opset,
        save_sidecar=not args.no_sidecar,
    )
    print(f"Exported ONNX model to {out_path}")

    if args.verify:
        from .onnx_export import OnnxTagger

        with OnnxTagger() as tagger:
            tagger.open(out_path)
            labels = tagger.labels()
            sample_out = tagger.tag([{}])
            print(
                f"Verified with onnxruntime: {len(labels)} labels, "
                f"dummy prediction={sample_out}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
