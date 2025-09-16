import argparse
import json
import os
from pathlib import Path

import yaml

from .train import train
from .evaluate import run_evaluation

_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_DIR = _ROOT / "config"
_RESULTS_DIR = _ROOT / ".research" / "iteration1"
_CHECKPOINT_FILE = _ROOT / "checkpoints" / "best_resnet18_cifar10.pth"


def _load_cfg(name: str):
    with open(_CONFIG_DIR / name, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _run(cfg_name: str):
    cfg = _load_cfg(cfg_name)
    cfg.update({
        "checkpoint_path": str(_CHECKPOINT_FILE),
        "data_root": cfg.get("data_root", "~/.cache/data"),
    })

    best_acc = train(cfg)
    print(json.dumps({"best_accuracy": best_acc}, indent=2))

    run_evaluation(str(_CHECKPOINT_FILE), cfg["batch_size"], str(_RESULTS_DIR))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SYMPHONY baseline experiment runner")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke-test", action="store_true", help="Run quick smoke test (1-epoch, 5k samples)")
    group.add_argument("--full-experiment", action="store_true", help="Run full CIFAR-10 training (100 epochs)")
    args = parser.parse_args()

    try:
        if args.smoke_test:
            _run("smoke_test.yaml")
        else:
            _run("full_experiment.yaml")
    except KeyboardInterrupt:
        print("\nInterrupted by user – exiting.")
