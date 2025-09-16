"""preprocess.py
Very small pre-processing stub – in a real experiment we would download
and clean datasets here.  For the smoke test we only need to generate
some synthetic data so that later stages can iterate over it.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np


def preprocess(config: Dict[str, Any]):
    """Return dummy token/label pairs.

    Returns
    -------
    processed_data
        Dict with keys "input" and "label" – both numpy arrays.
    """
    rng = np.random.default_rng(seed=config["general"]["seed"])

    n_samples = config["data"]["num_samples"]
    seq_len = config["data"].get("seq_len", 16)

    processed_data = {
        "input": rng.integers(low=0, high=1000, size=(n_samples, seq_len)),
        "label": rng.integers(low=0, high=1000, size=(n_samples, seq_len)),
    }

    return processed_data
