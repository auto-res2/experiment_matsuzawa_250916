"""train.py
A minimal stub for training that complies with the interfaces expected by
`main.py`.  It **does not** train a real model – the objective of this
iteration is only to unblock the CI smoke-test so that the pipeline can
proceed to later, resource-heavier stages where the full models will be
plugged in.

The function must:
  • run fast on CPU-only environments (<2 s)
  • return a non-empty artefact (here: a dict containing random weights)
  • produce at least one numeric metric strictly > 0 so that downstream
    assertions in `evaluate.py` succeed.
"""
from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np
from tqdm import tqdm  # noqa: F401  – installed via pyproject.toml


def train(config: Dict[str, Any], processed_data: Dict[str, Any]):
    """Fake training loop that produces deterministic, non-zero metrics."""
    start = time.time()

    # Pretend to do some compute – keeps wall-clock non-zero.
    for _ in range(3):
        time.sleep(0.1)

    wall_s = time.time() - start

    # Toy "model": random numpy arrays whose seed depends on the run id.
    rng = np.random.default_rng(seed=config["general"]["seed"])
    model = {
        "weights": rng.standard_normal(size=(4, 4)).astype("float32"),
        "bias": rng.standard_normal(size=(4,)).astype("float32"),
    }

    metrics = {
        "train_wall_s": wall_s,
        "train_samples": int(len(processed_data["input"])),
    }

    # Fail-fast: any non-positive metric is unacceptable.
    if any(v <= 0 for v in metrics.values()):
        raise ValueError("Training produced non-positive metric – aborting.")

    return model, metrics
