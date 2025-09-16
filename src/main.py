"""main.py – command-line entry-point orchestrating the workflow."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

# Relative imports – pylint/ruff might warn but this is intentional.
from .preprocess import load_dataset
from .train import save_model, train_model
from .evaluate import evaluate

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

SMOKE_YAML = CONFIG_DIR / "smoke_test.yaml"
FULL_YAML = CONFIG_DIR / "full_experiment.yaml"

HF_TOKEN = os.getenv("HF_TOKEN")  # Accessible to downstream loaders if needed.


def _parse_args() -> argparse.Namespace:  # noqa: D401
    p = argparse.ArgumentParser(description="AURORA Experiment Runner")

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke-test", action="store_true", help="Run quick validation only")
    mode.add_argument("--full-experiment", action="store_true", help="Run full-scale experiment")

    return p.parse_args()


def _load_yaml(path: Path) -> Dict[str, Any]:  # noqa: D401
    if not path.exists():
        sys.exit(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def main() -> None:  # noqa: D401
    args = _parse_args()

    config_path = SMOKE_YAML if args.smoke_test else FULL_YAML
    config = _load_yaml(config_path)

    # Phase 1 – Smoke Test ---------------------------------------------------
    print("[INFO] Loading dataset …", flush=True)
    dataset = load_dataset(config)

    print("[INFO] Training model …", flush=True)
    model = train_model(dataset, config)

    artifact_dir = Path(".research/iteration1/artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    save_model(model, artifact_dir / "model.pt")

    print("[INFO] Evaluating …", flush=True)
    metrics = evaluate(model, dataset, config)

    if args.smoke_test and not args.full_experiment:
        print("[INFO] Smoke test completed. Exiting.")
        return

    # Phase 2 – Full Experiment ---------------------------------------------
    if args.full_experiment:
        print("[INFO] ⚙️  Starting full experiment …", flush=True)
        # Potentially reload a larger dataset etc. For now we reuse the same.
        evaluate(model, dataset, config)


if __name__ == "__main__":  # pragma: no cover
    main()
