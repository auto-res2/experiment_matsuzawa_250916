#!/usr/bin/env python3
"""Entry-point orchestrating the (placeholder) SCARF pipeline.

Run with either ``--smoke-test`` or ``--full-experiment`` to pick the desired
configuration.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Dict, Any

# -----------------------------------------------------------------------------
# Make top-level project modules importable when ``src/`` is executed directly
# -----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# -----------------------------------------------------------------------------
# Local imports (deferred to ensure path fix above is in effect)
# -----------------------------------------------------------------------------
import yaml  # noqa: E402  pylint: disable=wrong-import-position

# We import the lightweight modules via ``importlib`` inside a helper to avoid
# import-time side-effects before CLI parsing is finished.

def _lazy_import(name: str):
    return importlib.import_module(name)


def _load_pipeline_modules():
    global preprocess, train, evaluate  # pylint: disable=global-statement
    preprocess = _lazy_import("preprocess_py").preprocess
    train = _lazy_import("train_py").train
    evaluate = _lazy_import("evaluate_py").evaluate


CONFIG_DIR = ROOT_DIR / "config"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _load_yaml(name: str) -> Dict[str, Any]:
    path = CONFIG_DIR / name
    if not path.exists():
        sys.exit(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SCARF experiment runner")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke-test", action="store_true", help="Run the smoke-test config")
    group.add_argument("--full-experiment", action="store_true", help="Run the full experiment config")
    return p.parse_args()


# -----------------------------------------------------------------------------
# Main orchestration
# -----------------------------------------------------------------------------

def main() -> None:  # pragma: no cover
    args = _parse_args()

    cfg = _load_yaml("smoke_test.yaml" if args.smoke_test else "full_experiment.yaml")

    # Import pipeline modules only after CLI args & config were loaded
    _load_pipeline_modules()

    # -------------------- pipeline stages --------------------
    preprocess(cfg)
    train_artifacts = train(cfg)
    evaluate(train_artifacts)


if __name__ == "__main__":
    main()
