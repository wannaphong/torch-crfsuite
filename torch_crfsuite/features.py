"""Feature extraction, ItemSequence, and vocabulary management."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from .crf1d import EncodedSequence

FeatureItem = Mapping[str, Any] | Sequence[str]


def flatten_item_features(item: FeatureItem, prefix: str = "") -> Dict[str, float]:
    """Convert a python-crfsuite item feature representation to {attr: weight}.

    Supported formats (matching ``scrapinghub/python-crfsuite``):
      - ``{"key": float_weight}``
      - ``{"key": bool}`` (``True`` -> 1.0, ``False`` -> 0.0)
      - ``{"key": "string_value"}`` -> ``{"key:string_value": 1.0}``
      - ``["key1", "key2"]`` -> ``{"key1": 1.0, "key2": 1.0}``
      - Nested dicts, lists, or sets with ``"prefix:"`` prepended to keys.
    """
    out: Dict[str, float] = {}

    if isinstance(item, Mapping):
        for k, v in item.items():
            key = f"{prefix}{k}"
            if isinstance(v, bool):
                if v:
                    out[key] = 1.0
            elif isinstance(v, (int, float)):
                val = float(v)
                if val != 0.0:
                    out[key] = val
            elif isinstance(v, str):
                out[f"{key}:{v}"] = 1.0
            elif isinstance(v, (Mapping, list, tuple, set, frozenset)):
                nested = flatten_item_features(
                    list(v) if isinstance(v, (set, frozenset)) else v,
                    prefix=f"{key}:",
                )
                out.update(nested)
            else:
                raise TypeError(
                    f"Unsupported feature value type {type(v).__name__!r} for key {k!r}"
                )
    elif isinstance(item, (list, tuple, set, frozenset)):
        for k in item:
            if not isinstance(k, str):
                raise TypeError(
                    f"List/set item features must be strings, got {type(k).__name__!r}"
                )
            out[f"{prefix}{k}"] = 1.0
    else:
        raise TypeError(
            f"Item features must be a dict/mapping or list/sequence of strings, "
            f"got {type(item).__name__!r}"
        )

    return out


class ItemSequence:
    """Wrapper for a sequence of item feature dicts/lists (``pycrfsuite.ItemSequence``)."""

    def __init__(self, xseq: Sequence[FeatureItem]) -> None:
        if isinstance(xseq, ItemSequence):
            self._items = list(xseq._items)
        else:
            self._items = [flatten_item_features(item) for item in xseq]

    def items(self) -> List[Dict[str, float]]:
        """Return processed attribute dicts for each position in the sequence."""
        return [dict(item) for item in self._items]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Dict[str, float]:
        return dict(self._items[idx])

    def __iter__(self):
        return iter(self._items)

    def __repr__(self) -> str:
        return f"<ItemSequence of length {len(self._items)}>"


def ensure_item_sequence(xseq: Sequence[FeatureItem] | ItemSequence) -> ItemSequence:
    if isinstance(xseq, ItemSequence):
        return xseq
    return ItemSequence(xseq)


class FeatureVocabulary:
    """Maps string attributes and labels to contiguous integer IDs."""

    def __init__(
        self,
        attr_to_id: Optional[Dict[str, int]] = None,
        label_to_id: Optional[Dict[str, int]] = None,
        state_mask: Optional[torch.Tensor] = None,
        trans_mask: Optional[torch.Tensor] = None,
    ) -> None:
        self.attr_to_id: Dict[str, int] = dict(attr_to_id or {})
        self.label_to_id: Dict[str, int] = dict(label_to_id or {})
        self.id_to_attr: List[str] = [""] * len(self.attr_to_id)
        for k, v in self.attr_to_id.items():
            self.id_to_attr[v] = k
        self.id_to_label: List[str] = [""] * len(self.label_to_id)
        for k, v in self.label_to_id.items():
            self.id_to_label[v] = k
        self.state_mask = state_mask
        self.trans_mask = trans_mask

    @classmethod
    def from_dataset(
        cls,
        sequences: Iterable[Tuple[ItemSequence, Sequence[str]]],
        minfreq: float = 0.0,
        possible_states: bool = False,
        possible_transitions: bool = False,
    ) -> "FeatureVocabulary":
        attr_counts: Counter[str] = Counter()
        label_set: Dict[str, int] = {}
        observed_states: set[Tuple[str, str]] = set()
        observed_transitions: set[Tuple[str, str]] = set()

        seq_list = list(sequences)
        for xseq, yseq in seq_list:
            for y in yseq:
                if y not in label_set:
                    label_set[y] = len(label_set)
            for t, (attrs, y) in enumerate(zip(xseq, yseq)):
                for attr in attrs:
                    attr_counts[attr] += 1
                    observed_states.add((attr, y))
                if t > 0:
                    observed_transitions.add((yseq[t - 1], y))

        attr_to_id: Dict[str, int] = {}
        for attr, count in attr_counts.items():
            if count >= minfreq:
                attr_to_id[attr] = len(attr_to_id)

        num_attrs = len(attr_to_id)
        num_labels = len(label_set)

        state_mask: Optional[torch.Tensor] = None
        if not possible_states and num_attrs > 0 and num_labels > 0:
            state_mask = torch.zeros((num_attrs, num_labels), dtype=torch.bool)
            for attr, y in observed_states:
                if attr in attr_to_id:
                    state_mask[attr_to_id[attr], label_set[y]] = True

        trans_mask: Optional[torch.Tensor] = None
        if not possible_transitions and num_labels > 0:
            trans_mask = torch.zeros((num_labels, num_labels), dtype=torch.bool)
            for y_prev, y_curr in observed_transitions:
                trans_mask[label_set[y_prev], label_set[y_curr]] = True

        return cls(
            attr_to_id=attr_to_id,
            label_to_id=label_set,
            state_mask=state_mask,
            trans_mask=trans_mask,
        )

    def encode_sequence(
        self,
        xseq: ItemSequence,
        yseq: Optional[Sequence[str]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> EncodedSequence:
        time_idx: List[int] = []
        attr_idx: List[int] = []
        attr_vals: List[float] = []

        for t, attrs in enumerate(xseq):
            for attr, val in attrs.items():
                aid = self.attr_to_id.get(attr)
                if aid is not None:
                    time_idx.append(t)
                    attr_idx.append(aid)
                    attr_vals.append(val)

        if time_idx:
            indices = torch.tensor([time_idx, attr_idx], dtype=torch.long)
            values = torch.tensor(attr_vals, dtype=dtype)
        else:
            indices = torch.zeros((2, 0), dtype=torch.long)
            values = torch.zeros((0,), dtype=dtype)

        label_ids: Optional[torch.Tensor] = None
        if yseq is not None:
            label_ids = torch.tensor(
                [self.label_to_id[y] for y in yseq], dtype=torch.long
            )

        return EncodedSequence(
            length=len(xseq),
            attr_indices=indices,
            attr_values=values,
            label_ids=label_ids,
        )
