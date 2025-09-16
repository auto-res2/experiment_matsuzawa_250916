"""preprocess.py – minimal data-loading helpers."""
from __future__ import annotations

import random
from typing import Any, Dict, List

__all__ = [
    "load_dataset",
]


_DUMMY_CORPUS: List[str] = [
    "the quick brown fox jumps over the lazy dog",
    "lorem ipsum dolor sit amet",
    "pack my box with five dozen liquor jugs",
]


def load_dataset(config: Dict[str, Any]) -> Dict[str, Any]:  # noqa: D401
    """Return a toy in-memory dataset so the experiment can run end-to-end."""
    sample_size = config.get("data", {}).get("sample_size", len(_DUMMY_CORPUS))
    rng = random.Random(config.get("seed", 42))
    samples = rng.sample(_DUMMY_CORPUS * ((sample_size // len(_DUMMY_CORPUS)) + 1), k=sample_size)  # noqa: E501

    return {"text": samples}
