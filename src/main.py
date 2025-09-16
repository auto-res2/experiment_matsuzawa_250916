import os
import argparse
import yaml
import logging
from datetime import datetime
import torch
import random
import numpy as np

from src.preprocess import prepare_all_data
from src.evaluate import run_experiments

# -----------------------------------------------------------------------------
# Global constants for mandatory save locations (iteration13)
# -----------------------------------------------------------------------------
BASE_RESEARCH_DIR = ".research/iteration13"
IMAGES_DIR = os.path.join(BASE_RESEARCH_DIR, "images")
RESULTS_DIR = os.path.join(BASE_RESEARCH_DIR, "results")
LOGS_DIR = os.path.join(BASE_RESEARCH_DIR, "logs")
JSON_DIR = BASE_RESEARCH_DIR  # JSON summaries are saved directly here

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logging.info(f"Global random seed set to {seed}")


def load_config(path: str) -> dict:
    logging.info(f"Loading configuration from: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found at {path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Run ZORRO++ experiments.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--smoke-test', action='store_true', help='Run a small-scale smoke test.')
    group.add_argument('--full-experiment', action='store_true', help='Run the full experiment suite.')
    args = parser.parse_args()

    if args.smoke_test:
        config_path = 'config/smoke_test.yaml'
        logging.info("Executing in SMOKE TEST mode.")
    else:
        config_path = 'config/full_experiment.yaml'
        logging.info("Executing in FULL EXPERIMENT mode.")

    # Load configuration
    config = load_config(config_path)
    set_seed(config['globals']['seed'])

    # Create output directories
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    config['globals']['base_dir'] = BASE_RESEARCH_DIR
    logging.info(f"Results will be saved in: {BASE_RESEARCH_DIR}")

    # Phase 1: Data Preparation
    logging.info("--- Phase 1: Data Preparation ---")
    try:
        prepare_all_data(config.get('data_manifest', {}))
        logging.info("All data assets are verified and ready.")
    except (FileNotFoundError, ValueError) as e:
        logging.error(f"Data preparation failed: {e}")
        logging.error("Aborting experiment. Please check data URLs, checksums, and network connection.")
        return

    # Phase 2: Experiment Execution and Evaluation
    logging.info("--- Phase 2: Running Experiments ---")
    try:
        run_experiments(config)
        logging.info("Experiment suite finished successfully.")
    except Exception as e:
        logging.error(f"An unhandled error occurred during experiment execution: {e}", exc_info=True)
        logging.error("Experiment run aborted.")


if __name__ == '__main__':
    main()