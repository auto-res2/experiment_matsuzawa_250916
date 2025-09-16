import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Use relative imports for sibling modules in the same package
from . import preprocess, train, evaluate

def setup_environment(config):
    """Creates directories and sets random seeds."""
    print("Setting up environment...")
    # Create directories
    for path in config['paths'].values():
        if path: # Ensure path is not None or empty
            Path(path).mkdir(parents=True, exist_ok=True)
    
    # Set seeds for reproducibility
    seed = config['seeds'][0] # Use the first seed for single runs
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

def main():
    """Main execution script for the experimental pipeline."""
    parser = argparse.ArgumentParser(description="Run REFLECT-BO experiments.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--smoke-test', action='store_true', help='Run a small-scale smoke test.')
    group.add_argument('--full-experiment', action='store_true', help='Run the full-scale experiment.')
    parser.add_argument('--experiment-id', type=int, choices=[1, 2, 3], required=True,
                        help='ID of the experiment to run (1, 2, or 3).')

    args = parser.parse_args()

    if args.smoke_test:
        config_path = 'config/smoke_test.yaml'
    else:
        config_path = 'config/full_experiment.yaml'

    print(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"ERROR: Configuration file not found at '{config_path}'")
        sys.exit(1)
    
    # Set which experiment to run
    config['experiment_to_run'] = args.experiment_id
    print(f"Selected to run Experiment {config['experiment_to_run']}")

    # --- Pipeline Execution ---
    try:
        # 1. Setup Environment
        setup_environment(config)

        # 2. Preprocessing
        print("\n--- STAGE 1: DATA PREPROCESSING ---")
        preprocessed_path = preprocess.run(config)
        print("--- PREPROCESSING COMPLETE ---")

        # 3. Training / Optimization
        print("\n--- STAGE 2: TRAINING / OPTIMIZATION ---")
        training_output_path = train.run(config, preprocessed_path)
        print("--- TRAINING COMPLETE ---")

        # 4. Evaluation
        print("\n--- STAGE 3: EVALUATION ---")
        evaluate.run(config, training_output_path)
        print("--- EVALUATION COMPLETE ---")

    except FileNotFoundError as e:
        print(f"\nERROR: A required file was not found.", file=sys.stderr)
        print(str(e), file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"\nERROR: A configuration or data value is invalid.", file=sys.stderr)
        print(str(e), file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {type(e).__name__}", file=sys.stderr)
        print(str(e), file=sys.stderr)
        sys.exit(1)

    print("\nExperiment pipeline finished successfully.")

if __name__ == '__main__':
    main()