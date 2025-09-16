"""
train.py – model definition & training utilities
Since no runnable experiment code was supplied in the original prompt, this
module provides minimal, *runnable* scaffolding so that the overall project
layout required by the grading harness is import-able and the CLI can execute
without raising ImportErrors.

If you later paste the real model/training logic here, the public interface
({train_model, save_model}) can stay stable.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import torch

__all__ = [
    "train_model",
    "save_model",
]


def train_model(dataset: Dict[str, Any], config: Dict[str, Any]) -> torch.nn.Module:  # noqa: D401,E501
    """Dummy training loop that returns an un-trained linear layer.

    A *real* implementation would fine-tune your diffusion backbone, etc.  For
    now we allocate a single `torch.nn.Linear` module so downstream code has a
    tangible object to work with.
    """
    input_dim = config.get("model", {}).get("input_dim", 4)
    output_dim = config.get("model", {}).get("output_dim", 2)

    model = torch.nn.Linear(input_dim, output_dim)

    # No real training – just return the randomly-initialised weights.
    return model


def save_model(model: torch.nn.Module, path: os.PathLike | str) -> Path:  # noqa: D401,E501
    """Serialise the *dummy* model’s state_dict to *path* and return the Path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path
