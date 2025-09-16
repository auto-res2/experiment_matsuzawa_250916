import argparse
import os
import sys
import yaml
import random
import numpy as np
import torch

# Ensure src is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import preprocess, train, evaluate  # noqa: E402


# -----------------------------------------------------------------------------
#  Utility helpers
# -----------------------------------------------------------------------------

def set_seeds(seed_value: int):
    """Make experiment deterministic so results are reproducible."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(path: str) -> dict:
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{path}'.", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error: Could not parse YAML file '{path}': {e}", file=sys.stderr)
        sys.exit(1)


# -----------------------------------------------------------------------------
#  Stage runner
# -----------------------------------------------------------------------------

def run_stage(stage: str, config: dict):
    print(f"\n{'='*20} RUNNING STAGE: {stage.upper()} {'='*20}")
    if stage == "preprocess":
        preprocess.run(config)
    elif stage == "train":
        train.run(config)
    elif stage == "evaluate":
        evaluate.run(config)
    else:
        raise ValueError(
            "Unknown stage: '{stage}'. Must be one of 'preprocess', 'train', 'evaluate'."
        )
    print(f"{'='*20} STAGE COMPLETE: {stage.upper()} {'='*20}")


# -----------------------------------------------------------------------------
#  Main CLI entry-point
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run REFLECT-BO research pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Positional argument *stage* is now optional; default runs the full pipeline
    parser.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=["preprocess", "train", "evaluate", "all"],
        help="Pipeline stage to execute. If omitted, the full pipeline runs.",
    )

    # Configuration selection (mutually exclusive)
    cfg_group = parser.add_mutually_exclusive_group(required=True)
    cfg_group.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run the lightweight smoke-test configuration.",
    )
    cfg_group.add_argument(
        "--full-experiment",
        action="store_true",
        help="Run the full experimental configuration.",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    #  Load YAML config
    # ------------------------------------------------------------------
    if args.smoke_test:
        cfg_path = "config/smoke_test.yaml"
        print("--- Mode: SMOKE TEST ---")
    else:
        cfg_path = "config/full_experiment.yaml"
        print("--- Mode: FULL EXPERIMENT ---")

    config = load_config(cfg_path)
    set_seeds(config["seeds"][0])  # Global seed for reproducibility

    # For robustness, run a quick smoke-test before the full experiment
    if args.full_experiment:
        print("\n--- Validating pipeline with a smoke test before full run… ---")
        smoke_cfg = load_config("config/smoke_test.yaml")
        try:
            set_seeds(smoke_cfg["seeds"][0])
            smoke_cfg["experiment_id"] = smoke_cfg["experiment_ids_to_run"][0]
            run_stage("preprocess", smoke_cfg)
            run_stage("train", smoke_cfg)
            run_stage("evaluate", smoke_cfg)
            print("--- Smoke test PASSED. Proceeding to full experiment. ---")
        except Exception as e:
            print(
                f"\nFATAL: Smoke test FAILED. Aborting full experiment. Error: {e}",
                file=sys.stderr,
            )
            import traceback

            traceback.print_exc()
            sys.exit(1)
        # Reset RNG for main experiment
        set_seeds(config["seeds"][0])

    # ------------------------------------------------------------------
    #  Execute requested stage(s)
    # ------------------------------------------------------------------
    stages_to_run = (
        ["preprocess", "train", "evaluate"] if args.stage == "all" else [args.stage]
    )

    try:
        for exp_id in config["experiment_ids_to_run"]:
            print(
                f"\n{'#'*60}\n# Experiment ID: {exp_id:<3}{'#':>40}\n{'#'*60}"
            )
            config["experiment_id"] = exp_id  # inject current experiment ID
            for stage in stages_to_run:
                run_stage(stage, config)
    except (
        FileNotFoundError,
        ConnectionError,
        RuntimeError,
        ValueError,
        AssertionError,
    ) as e:
        print(f"\nFATAL ERROR during experiment execution: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nUNHANDLED EXCEPTION: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("\nAll specified experiments and stages completed successfully.")


if __name__ == "__main__":
    main()
