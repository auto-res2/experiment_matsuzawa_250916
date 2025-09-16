import os
import argparse
import yaml
import logging
from datetime import datetime
import torch
import wandb
import random
import numpy as np

from .preprocess import get_dataloaders
from .train import run_experiment, make_run_name
from .evaluate import generate_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path):
    logger.info(f"Loading config {path}")
    with open(path, 'r') as fh:
        return yaml.safe_load(fh)


def main():
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--smoke-test', action='store_true')
    group.add_argument('--full-experiment', action='store_true')
    args = ap.parse_args()

    cfg_path = 'config/smoke_test.yaml' if args.smoke_test else 'config/full_experiment.yaml'
    config = load_config(cfg_path)

    ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # Updated to iteration9 directory structure (spec requirement)
    results_dir = os.path.join('.research', 'iteration9', f'results_{ts}')
    os.makedirs(results_dir, exist_ok=True)
    config['results_dir'] = results_dir

    if config.get('wandb', {}).get('enabled'):
        try:
            wandb.init(project=config['wandb']['project'], entity=config['wandb'].get('entity'), config=config)
        except Exception as e:
            logger.warning(f"wandb initialisation failed: {e}. Disabling.")
            config['wandb']['enabled'] = False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device {device}")

    logger.info("Preparing dataloaders …")
    dataloaders = get_dataloaders(config)

    logger.info("Launching experiments …")
    for exp_cfg in config['experiments']:
        # skip template configs
        if 'corruption' not in exp_cfg['dataset']:
            continue
        try:
            run_name = make_run_name(exp_cfg)
        except ValueError:
            logger.debug("Template encountered inside loop – skipping.")
            continue
        if run_name not in dataloaders:
            logger.warning(f"No dataloader for {run_name}. Skipping run.")
            continue
        logger.info(f"---- Running {run_name} ----")
        set_seed(exp_cfg['seed'])
        exp_cfg = {**config, **exp_cfg, 'results_dir': results_dir}  # merge
        try:
            run_experiment(exp_cfg, dataloaders[run_name], device)
        except Exception as e:
            logger.error(f"Run {run_name} failed: {e}", exc_info=True)

    logger.info("All runs complete. Generating report …")
    generate_report(results_dir)

    if config.get('wandb', {}).get('enabled'):
        wandb.finish()

    logger.info("Workflow finished.")

if __name__ == '__main__':
    main()