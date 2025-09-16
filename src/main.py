import argparse
import yaml
import os
import torch
import numpy as np

# Use relative imports for sibling modules in the same package
from . import preprocess
from . import train
from . import evaluate

def set_global_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description="Run META-LEAP experiments.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--smoke-test', action='store_true', help='Run a small-scale smoke test.')
    group.add_argument('--full-experiment', action='store_true', help='Run the full experiment.')
    
    args = parser.parse_args()

    if args.smoke_test:
        config_path = 'config/smoke_test.yaml'
    else:
        config_path = 'config/full_experiment.yaml'

    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if args.smoke_test:
        config['is_smoke_test'] = True

    # Setup directories
    os.makedirs('.research/iteration1', exist_ok=True)
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)

    # Set global seed for reproducibility
    set_global_seed(config['global_seed'])
    print(f"Global seed set to {config['global_seed']}")

    # --- Stage 1: Preprocessing ---
    print("\n--- Running Preprocessing Stage ---")
    try:
        preprocess.main(config)
        print("--- Preprocessing Stage Complete ---")
    except FileNotFoundError as e:
        print(f"\nERROR in preprocessing: {e}")
        print("This is expected for the full experiment if you haven't downloaded the custom datasets.")
        print("Please place the required data files in 'data/raw/' and try again.")
        print("Aborting experiment.")
        return

    # --- Stage 2: Training ---
    print("\n--- Running Training Stage ---")
    train.main(config)
    print("--- Training Stage Complete ---")

    # --- Stage 3: Evaluation ---
    print("\n--- Running Evaluation Stage ---")
    evaluate.main(config)
    print("--- Evaluation Stage Complete ---")

if __name__ == '__main__':
    main()
