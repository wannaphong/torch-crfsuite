"""Tests for torch_crfsuite (PyTorch rewrite of crfsuite and python-crfsuite)."""

from __future__ import annotations

import io
import itertools
import math
from pathlib import Path

import pytest
import torch

from torch_crfsuite import ItemSequence, LinearChainCRF, OnnxTagger, Tagger, Trainer
from torch_crfsuite.crf1d import EncodedSequence


def _sample_dataset():
    """Small synthetic POS/chunking dataset for end-to-end training tests."""
    return [
        (
            [
                {"word": "the", "pos": "DT", "bos": True},
                {"word": "cat", "pos": "NN", "cap": False},
                {"word": "sat", "pos": "VBD", "cap": False},
            ],
            ["B-NP", "I-NP", "B-VP"],
        ),
        (
            [
                {"word": "a", "pos": "DT", "bos": True},
                {"word": "dog", "pos": "NN", "cap": False},
                {"word": "ran", "pos": "VBD", "cap": False},
            ],
            ["B-NP", "I-NP", "B-VP"],
        ),
        (
            [
                {"word": "dogs", "pos": "NNS", "bos": True},
                {"word": "bark", "pos": "VBP", "cap": False},
            ],
            ["B-NP", "B-VP"],
        ),
        (
            [
                {"word": "cats", "pos": "NNS", "bos": True},
                {"word": "sleep", "pos": "VBP", "cap": False},
            ],
            ["B-NP", "B-VP"],
        ),
    ]


def test_item_sequence_formats():
    raw = [
        {
            "w": "Hello",
            "is_cap": True,
            "is_num": False,
            "len": 5,
            "bias": 1.0,
            "prev": {"w": "<BOS>", "tags": ["BOS", "START"]},
            "suffixes": {"lo", "llo"},
        },
        ["word=world", "lower"],
    ]
    seq = ItemSequence(raw)
    items = seq.items()
    assert len(seq) == 2
    assert items[0]["w:Hello"] == 1.0
    assert items[0]["is_cap"] == 1.0
    assert "is_num" not in items[0]
    assert items[0]["len"] == 5.0
    assert items[0]["prev:w:<BOS>"] == 1.0
    assert items[0]["prev:tags:BOS"] == 1.0
    assert items[0]["suffixes:lo"] == 1.0
    assert items[1] == {"word=world": 1.0, "lower": 1.0}


def test_exact_partition_marginals_and_viterbi():
    torch.manual_seed(42)
    crf = LinearChainCRF(num_attributes=4, num_labels=3, dtype=torch.float64)
    with torch.no_grad():
        crf.state_weights.normal_(0.0, 0.7)
        crf.trans_weights.normal_(0.0, 0.5)

    seq = EncodedSequence(
        length=3,
        attr_indices=torch.tensor([[0, 0, 1, 2, 2], [0, 2, 1, 0, 3]], dtype=torch.long),
        attr_values=torch.tensor([1.0, 0.5, 1.2, -0.4, 0.8], dtype=torch.float64),
    )

    emissions = crf.compute_emissions(seq)
    _, log_z = crf.forward_alphas(emissions)
    marginals = crf.marginals(seq)
    viterbi_path, viterbi_score = crf.viterbi_decode(seq)

    all_scores = []
    best_bf_score = -float("inf")
    best_bf_path = None
    for path in itertools.product(range(3), repeat=3):
        y = torch.tensor(path, dtype=torch.long)
        s = float(crf.sequence_score(emissions, y).item())
        all_scores.append(s)
        if s > best_bf_score:
            best_bf_score = s
            best_bf_path = list(path)

    bf_log_z = math.log(sum(math.exp(s) for s in all_scores))
    assert math.isclose(float(log_z.item()), bf_log_z, rel_tol=1e-6, abs_tol=1e-6)
    assert viterbi_path == best_bf_path
    assert math.isclose(viterbi_score, best_bf_score, rel_tol=1e-6, abs_tol=1e-6)

    row_sums = marginals.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(3, dtype=torch.float64), atol=1e-6)


@pytest.mark.parametrize("algorithm", ["lbfgs", "l2sgd", "ap", "pa", "arow"])
def test_all_training_algorithms(tmp_path: Path, algorithm: str):
    data = _sample_dataset()
    trainer = Trainer(algorithm=algorithm, verbose=False)
    trainer.set("max_iterations", 25)
    if algorithm in ("lbfgs", "l2sgd"):
        trainer.set("c2", 0.1)

    for xseq, yseq in data:
        trainer.append(xseq, yseq)

    model_file = tmp_path / f"model_{algorithm}.pt"
    trainer.train(model_file)
    assert model_file.exists()

    with Tagger() as tagger:
        tagger.open(model_file)
        assert set(tagger.labels()) == {"B-NP", "I-NP", "B-VP"}

        test_x = [
            {"word": "the", "pos": "DT", "bos": True},
            {"word": "cat", "pos": "NN", "cap": False},
            {"word": "sat", "pos": "VBD", "cap": False},
        ]
        pred = tagger.tag(test_x)
        assert pred == ["B-NP", "I-NP", "B-VP"]

        prob = tagger.probability(pred)
        assert 0.0 < prob <= 1.0

        marg_sum = sum(tagger.marginal(lbl, 0) for lbl in tagger.labels())
        assert math.isclose(marg_sum, 1.0, abs_tol=1e-5)


def test_open_inmemory_info_dump_and_holdout(tmp_path: Path):
    data = _sample_dataset()
    trainer = Trainer(algorithm="lbfgs", verbose=False)
    trainer.set_params({"c1": 0.0, "c2": 0.5, "max_iterations": 20, "feature.minfreq": 1})
    for idx, (xseq, yseq) in enumerate(data):
        trainer.append(xseq, yseq, group=1 if idx == len(data) - 1 else 0)

    model_file = tmp_path / "model.pt"
    trainer.train(model_file, holdout=1)

    raw_bytes = model_file.read_bytes()
    tagger = Tagger()
    tagger.open_inmemory(raw_bytes)

    info = tagger.info()
    assert len(info.labels) == 3
    assert len(info.state_features) > 0
    assert len(info.transitions) > 0

    buf = io.StringIO()
    tagger.dump(buf)
    dumped = buf.getvalue()
    assert "LABELS =" in dumped
    assert "TRANSITIONS =" in dumped
    assert "STATE_FEATURES =" in dumped


def test_onnx_export_and_onnxruntime_inference(tmp_path: Path):
    pytest.importorskip("onnxruntime")

    data = _sample_dataset()
    trainer = Trainer(algorithm="lbfgs", verbose=False)
    trainer.set_params({"c2": 0.1, "max_iterations": 25})
    for xseq, yseq in data:
        trainer.append(xseq, yseq)

    pt_path = tmp_path / "model.pt"
    onnx_path = tmp_path / "model.onnx"
    trainer.train(pt_path)

    test_x = [
        {"word": "the", "pos": "DT", "bos": True},
        {"word": "cat", "pos": "NN", "cap": False},
        {"word": "sat", "pos": "VBD", "cap": False},
    ]

    with Tagger() as pt_tagger:
        pt_tagger.open(pt_path)
        pt_tagger.to_onnx(onnx_path)
        pt_pred = pt_tagger.tag(test_x)
        pt_prob = pt_tagger.probability(pt_pred)
        pt_margs = [pt_tagger.marginal(lbl, 0) for lbl in pt_tagger.labels()]

    assert onnx_path.exists()

    with OnnxTagger() as ort_tagger:
        ort_tagger.open(onnx_path)
        ort_pred = ort_tagger.tag(test_x)
        ort_prob = ort_tagger.probability(ort_pred)
        ort_margs = [ort_tagger.marginal(lbl, 0) for lbl in ort_tagger.labels()]

    assert ort_pred == pt_pred
    assert math.isclose(ort_prob, pt_prob, rel_tol=1e-4, abs_tol=1e-4)
    for pm, om in zip(pt_margs, ort_margs):
        assert math.isclose(pm, om, rel_tol=1e-4, abs_tol=1e-4)


def test_convert_crfsuite_dump_to_onnx(tmp_path: Path):
    pytest.importorskip("onnxruntime")
    from torch_crfsuite import convert_crfsuite_to_onnx

    dump_text = """FILEHEADER = {
  magic: lCRF
  num_labels: 2
}

LABELS = {
  0: B
  1: I
}

ATTRIBUTES = {
  0: c[0]=ก
  1: c[0]=า
}

TRANSITIONS = {
  (1) B --> I: 2.000000
  (1) I --> B: 1.500000
}

STATE_FEATURES = {
  (0) c[0]=ก --> B: 3.000000
  (0) c[0]=า --> I: 2.500000
}
"""
    dump_path = tmp_path / "sample.dump"
    onnx_path = tmp_path / "sample.onnx"
    dump_path.write_text(dump_text, encoding="utf-8")

    convert_crfsuite_to_onnx(dump_path, onnx_path)
    assert onnx_path.exists()

    with OnnxTagger() as tagger:
        tagger.open(onnx_path)
        assert tagger.labels() == ["B", "I"]
        pred = tagger.tag([{"c[0]=ก": 1.0}, {"c[0]=า": 1.0}])
        assert pred == ["B", "I"]
