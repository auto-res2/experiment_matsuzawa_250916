import argparse
import os
import sys
import random
from pathlib import Path
import yaml
import torch
import numpy as np

# Add src to path to allow relative imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import preprocess
import train
import evaluate

def set_global_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Global random seed set to {seed}")

def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"FATAL: Configuration file not found at '{config_path}'", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"FATAL: Error parsing YAML file '{config_path}': {e}", file=sys.stderr)
        sys.exit(1)

def _ensure_experiment_id_in_config(cfg):
    """Utility to guarantee that `cfg['experiment_id']` exists.

    Many helper functions assume the key is present. During smoke-tests we only
    have `experiment_ids_to_run`, so we derive the first (and only) element
    from that list unless already provided."""
    if 'experiment_id' not in cfg or cfg['experiment_id'] is None:
        ids = cfg.get('experiment_ids_to_run', [])
        if not ids:
            raise KeyError("Configuration is missing both 'experiment_id' and 'experiment_ids_to_run'.")
        cfg['experiment_id'] = ids[0]


def main():
    parser = argparse.ArgumentParser(description="Run REFLECT-BO experiments end-to-end.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--smoke-test', action='store_true', help='Run a small-scale smoke test.')
    group.add_argument('--full-experiment', action='store_true', help='Run the full-scale experiment.')

    args = parser.parse_args()

    if args.smoke_test:
        config_path = 'config/smoke_test.yaml'
        print("Executing SMOKE TEST...")
    else:
        config_path = 'config/full_experiment.yaml'
        print("Executing FULL EXPERIMENT...")

    config = load_config(config_path)
    set_global_seeds(config['seeds'][0])  # Use first seed for global setup

    # Phase 1: Two-Phase Execution (Smoke Test then Full Experiment)
    if args.full_experiment:
        print("\nPHASE 1: Running smoke test first for validation...")
        smoke_config = load_config('config/smoke_test.yaml')
        _ensure_experiment_id_in_config(smoke_config)
        set_global_seeds(smoke_config['seeds'][0])
        try:
            # A minimal run to validate the pipeline
            temp_path = preprocess.run_preprocessing(smoke_config)
            temp_train_dir = train.run_experiment(smoke_config, temp_path)
            evaluate.run_evaluation(smoke_config, temp_train_dir)
            print("\nSmoke test PASSED. Proceeding to full experiment.")
        except Exception as e:
            print(f"\nFATAL: Smoke test FAILED with error: {e}", file=sys.stderr)
            print("Aborting full experiment to save resources.", file=sys.stderr)
            sys.exit(1)

    # Reset seed for the main experiment run
    _ensure_experiment_id_in_config(config)  # ensures key exists before loops
    set_global_seeds(config['seeds'][0])

    # PHASE 2: Main Experiment Pipeline
    try:
        for exp_id in config['experiment_ids_to_run']:
            print("\n=======================================================")
            print(f"            STARTING EXPERIMENT {exp_id}")
            print("=======================================================")
            config['experiment_id'] = exp_id  # inject current id into config

            # 1. Data Preprocessing
            print("\n--- STAGE 1: DATA PREPROCESSING ---")
            processed_data_path = preprocess.run_preprocessing(config)
            print("--- PREPROCESSING COMPLETE ---")

            # 2. Training / Optimization
            print("\n--- STAGE 2: TRAINING / OPTIMIZATION ---")
            training_run_dir = train.run_experiment(config, processed_data_path)
            print("--- TRAINING COMPLETE ---")

            # 3. Evaluation
            print("\n--- STAGE 3: EVALUATION ---")
            evaluate.run_evaluation(config, training_run_dir)
            print("--- EVALUATION COMPLETE ---")

    except (FileNotFoundError, ConnectionError, RuntimeError, ValueError) as e:
        print(f"\nFATAL ERROR in pipeline: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"\nUNHANDLED EXCEPTION: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    print("\nAll configured experiments finished successfully.")

if __name__ == '__main__':
    main()