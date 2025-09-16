import argparse
import os
import sys
import yaml
import random
import numpy as np
import torch

# Ensure src is in the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import preprocess, train, evaluate

def set_seeds(seed_value):
    """Set seeds for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_config(path: str) -> dict:
    """Load a YAML configuration file."""
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{path}'.", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error: Could not parse YAML file '{path}': {e}", file=sys.stderr)
        sys.exit(1)

def run_stage(stage: str, config: dict):
    """Run a specific stage of the experimental pipeline."""
    print(f"\n{'='*20} RUNNING STAGE: {stage.upper()} {'='*20}")
    if stage == 'preprocess':
        preprocess.run(config)
    elif stage == 'train':
        train.run(config)
    elif stage == 'evaluate':
        evaluate.run(config)
    else:
        raise ValueError(f"Unknown stage: '{stage}'. Must be one of 'preprocess', 'train', 'evaluate'.")
    print(f"{'='*20} STAGE COMPLETE: {stage.upper()} {'='*20}")

def main():
    parser = argparse.ArgumentParser(description="Run REFLECT-BO research experiments.")
    
    # Stage selection
    parser.add_argument(
        'stage',
        type=str,
        choices=['preprocess', 'train', 'evaluate', 'all'],
        help="The experimental stage to run."
    )

    # Configuration selection (mutually exclusive)
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument(
        '--smoke-test',
        action='store_true',
        help="Run with the smoke test configuration for quick validation."
    )
    config_group.add_argument(
        '--full-experiment',
        action='store_true',
        help="Run with the full experiment configuration for publication-worthy results."
    )

    args = parser.parse_args()

    # Determine which config file to load
    if args.smoke_test:
        config_path = 'config/smoke_test.yaml'
        print("--- Mode: SMOKE TEST ---")
    else:
        config_path = 'config/full_experiment.yaml'
        print("--- Mode: FULL EXPERIMENT ---")

    # Load the primary configuration
    config = load_config(config_path)
    set_seeds(config['seeds'][0]) # Use the first seed for global setup
    
    # For full experiment, first run a validation smoke test
    if args.full_experiment:
        print("\n--- Validating pipeline with a smoke test before full run... ---")
        smoke_config = load_config('config/smoke_test.yaml')
        try:
            set_seeds(smoke_config['seeds'][0])
            # Run smoke test for the first experiment ID only
            smoke_config['experiment_id'] = smoke_config['experiment_ids_to_run'][0]
            run_stage('preprocess', smoke_config)
            run_stage('train', smoke_config)
            run_stage('evaluate', smoke_config)
            print("--- Smoke test PASSED. Proceeding to full experiment. ---")
        except Exception as e:
            print(f"\nFATAL: Smoke test FAILED. Aborting full experiment. Error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)
        # Reset seed for the main run
        set_seeds(config['seeds'][0])

    # Execute the requested stage(s)
    stages_to_run = ['preprocess', 'train', 'evaluate'] if args.stage == 'all' else [args.stage]
    
    try:
        for exp_id in config['experiment_ids_to_run']:
            print(f"\n{'#'*60}\n{'#':<2} Experiment ID: {exp_id}{'#':>40}\n{'#'*60}")
            # Inject the current experiment ID into the config for this run
            config['experiment_id'] = exp_id
            for stage in stages_to_run:
                run_stage(stage, config)
    except (FileNotFoundError, ConnectionError, RuntimeError, ValueError, AssertionError) as e:
        print(f"\nFATAL ERROR during experiment execution: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nUNHANDLED EXCEPTION: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\nAll specified experiments and stages completed successfully.")

if __name__ == '__main__':
    main()
