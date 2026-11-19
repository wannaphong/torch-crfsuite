"""Pure-PyTorch rewrite of CRFsuite and python-crfsuite."""

from .api import BaseTrainer, ParsedDump, Tagger, Trainer
from .convert_crfsuite import (
    convert_crfsuite_to_onnx,
    load_crfsuite_model,
    parse_crfsuite_dump,
)
from .crf1d import EncodedSequence, LinearChainCRF
from .features import FeatureVocabulary, ItemSequence
from .onnx_export import ONNXCRFModule, OnnxTagger, export_to_onnx

__all__ = [
    "BaseTrainer",
    "EncodedSequence",
    "FeatureVocabulary",
    "ItemSequence",
    "LinearChainCRF",
    "ONNXCRFModule",
    "OnnxTagger",
    "ParsedDump",
    "Tagger",
    "Trainer",
    "convert_crfsuite_to_onnx",
    "export_to_onnx",
    "load_crfsuite_model",
    "parse_crfsuite_dump",
]
__version__ = "0.1.0"
