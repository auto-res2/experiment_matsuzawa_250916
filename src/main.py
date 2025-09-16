import os
import argparse
import yaml
import logging
from datetime import datetime
import torch
import wandb
import random
import numpy as np

# Use relative imports for custom modules
from .preprocess import get_dataloaders
from .train import run_experiment
from .evaluate import generate_report

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path):
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file: {e}")
            raise
    return config

def main():
    parser = argparse.ArgumentParser(description="Run ZORRO++ experiments.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--smoke-test', action='store_true', help='Run a small-scale smoke test.')
    group.add_argument('--full-experiment', action='store_true', help='Run the full experiment.')
    args = parser.parse_args()

    if args.smoke_test:
        config_path = 'config/smoke_test.yaml'
    else:
        config_path = 'config/full_experiment.yaml'

    # --- 1. Setup --- 
    config = load_config(config_path)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results_dir = os.path.join('.research/iteration1', f'results_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    config['results_dir'] = results_dir
    
    # Init WandB
    if config.get('wandb', {}).get('enabled', False):
        try:
            wandb.init(
                project=config['wandb']['project'],
                entity=config['wandb'].get('entity'),
                config=config
            )
        except Exception as e:
            logger.warning(f"Could not initialize wandb: {e}. Disabling wandb.")
            config['wandb']['enabled'] = False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # --- 2. Data Preparation ---
    logger.info("Preparing dataloaders...")
    # The config structure from YAML might need to be massaged for get_dataloaders
    # For now, we assume a flat list of experiments in the config.
    dataloaders = get_dataloaders(config)
    if not dataloaders:
        logger.error("No dataloaders were created. Please check the dataset configurations.")
        return

    # --- 3. Run Experiments ---
    logger.info("Starting experimental runs...")
    experiment_configs = config.get('experiments', [])
    for exp_config in experiment_configs:
        run_name = f"{exp_config['exp_name']}_{exp_config['model']['name']}_{exp_config['dataset']['name']}_{exp_config['dataset']['corruption']}_{exp_config['method']}_eta{exp_config['stream']['eta']}_seed{exp_config['seed']}"
        if run_name in dataloaders:
            logger.info(f"--- Running: {run_name} ---")
            set_seed(exp_config['seed'])
            # Merge global config with run-specific config
            run_config = {**config, **exp_config}
            run_config['results_dir'] = results_dir # ensure path is correct
            try:
                run_experiment(run_config, dataloaders[run_name], device)
            except Exception as e:
                logger.error(f"Experiment {run_name} failed with error: {e}", exc_info=True)
        else:
            logger.warning(f"Dataloader for run {run_name} not found. Skipping.")

    # --- 4. Evaluation --- 
    logger.info("All experiments finished. Generating final report.")
    generate_report(results_dir)

    if config.get('wandb', {}).get('enabled', False):
        wandb.finish()

    logger.info("Workflow complete.")

if __name__ == '__main__':
    main()
