"""Main entry-point – orchestrates preprocessing, training & evaluation.

Usage
-----
uv run python -m src.main --smoke-test
uv run python -m src.main --full-experiment
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml

from . import preprocess, train, evaluate

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ARTIFACT_DIR = _REPO_ROOT / ".research" / "iteration8"
_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _run(cfg: Dict[str, Any], tag: str):
    # 1. Data -----------------------------------------------------------------
    loader, vocab = preprocess.build_dataloaders(cfg)

    # 2. Train ----------------------------------------------------------------
    model, train_stats = train.train_model(loader, vocab, cfg)

    # 3. Evaluate -------------------------------------------------------------
    eval_stats = evaluate.evaluate_model(model, loader, cfg)

    result = {"config_tag": tag, **train_stats, **eval_stats}

    # 4. Persist --------------------------------------------------------------
    out_file = _ARTIFACT_DIR / f"{tag}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    # Print for CI visibility
    print(json.dumps(result, indent=2))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke-test", action="store_true", help="run quick validation pipeline")
    mode.add_argument("--full-experiment", action="store_true", help="run full-scale experiment")
    args = parser.parse_args()

    if args.smoke_test:
        cfg = _load_config(_REPO_ROOT / "config" / "smoke_test.yaml")
        _run(cfg, "smoke_test")
    else:  # full experiment – only executed if the quick test succeeds in CI.
        cfg = _load_config(_REPO_ROOT / "config" / "full_experiment.yaml")
        _run(cfg, "full_experiment")


if __name__ == "__main__":  # pragma: no cover
    main()
