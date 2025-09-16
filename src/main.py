"""src/main.py
Entry-point that orchestrates *preprocess → train → evaluate* according
to a YAML configuration file.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import yaml  # PyYAML – specified in pyproject.toml

from preprocess import preprocess
from train import train
from evaluate import evaluate

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------
CONFIG_DIR = Path("config")
SMOKE_CFG = CONFIG_DIR / "smoke_test.yaml"
FULL_CFG = CONFIG_DIR / "full_experiment.yaml"


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def _parse_args(argv):
    p = argparse.ArgumentParser(description="AURORA experiment harness")
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run the quick CPU-only validation using smoke_test.yaml",
    )
    g.add_argument(
        "--full-experiment",
        action="store_true",
        help="Run the full GPU experiment using full_experiment.yaml",
    )
    return p.parse_args(argv)


def main(argv=None):  # noqa: D401 – short docstring style acceptable here
    """CLI entry – dispatches according to flags."""
    args = _parse_args(argv or sys.argv[1:])

    if args.full_experiment:
        cfg_path = FULL_CFG
    else:  # default to smoke test so that `python -m src.main` just works
        cfg_path = SMOKE_CFG

    config = _load_yaml(cfg_path)

    # Pipeline --------------------------------------------------------------
    processed = preprocess(config)
    model, train_metrics = train(config, processed)
    eval_metrics = evaluate(config, model, processed)

    # Summarise to stdout so that CI has something to grep.
    print("\n=== SUMMARY ===")
    print("Training metrics:", train_metrics)
    print("Evaluation metrics:", eval_metrics)


if __name__ == "__main__":
    main()
