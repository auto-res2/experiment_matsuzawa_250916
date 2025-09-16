#!/usr/bin/env python3
"""Entry-point orchestrating the (placeholder) SCARF pipeline.

Usage
-----
python main.py --smoke-test       # quick CI run
python main.py --full-experiment  # long run (still simulated)
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, Any

import yaml

from preprocess_py import preprocess
from train_py import train
from evaluate_py import evaluate

CONFIG_DIR = Path("config")


def _load_yaml(name: str) -> Dict[str, Any]:
    path = CONFIG_DIR / name
    if not path.exists():
        sys.exit(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SCARF experiment runner")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke-test", action="store_true", help="Run the smoke-test config")
    group.add_argument("--full-experiment", action="store_true", help="Run the full experiment config")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.smoke_test:
        cfg = _load_yaml("smoke_test.yaml")
    else:  # --full-experiment
        cfg = _load_yaml("full_experiment.yaml")

    # Pipeline ────────────────────────────────────────────────────────────
    preprocess(cfg)
    train_artifacts = train(cfg)
    evaluate(train_artifacts)


if __name__ == "__main__":
    main()
