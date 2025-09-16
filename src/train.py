import os
import time
import logging
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.stats import t
import timm
from transformers import Wav2Vec2Model
import pynvml
from fvcore.nn import FlopCountAnalysis
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

################################################################################
# Helper: consistent run-name generator (shared with main.py & preprocess.py)
################################################################################

def make_run_name(exp_cfg: dict) -> str:
    """Create a unique, consistent run identifier across the code-base.

    Templates (those experiment entries that miss the `corruption` field) will
    raise a ``ValueError`` so the caller can decide to skip them.
    """
    if 'corruption' not in exp_cfg['dataset']:
        raise ValueError("Config appears to be a template ‑ no `corruption` key.")

    corr   = exp_cfg['dataset']['corruption']
    eta    = exp_cfg['stream']['eta'] if 'stream' in exp_cfg else 'na'
    method = exp_cfg.get('method', 'na')

    return (
        f"{exp_cfg['exp_name']}_"
        f"{exp_cfg['model']['name']}_"
        f"{exp_cfg['dataset']['name']}_"
        f"{corr}_{method}_"
        f"eta{eta}_seed{exp_cfg['seed']}"
    )

# --- Helper Functions & Classes ------------------------------------------------

def get_feature_extractor(model, model_name):
    if 'resnet' in model_name or 'mobilenet' in model_name:
        feature_extractor = nn.Sequential(*list(model.children())[:-1])
    elif 'vit' in model_name:
        feature_extractor = model  # ViT exposes forward_features
    elif 'wav2vec' in model_name:
        feature_extractor = model
    else:
        raise ValueError(f"Unknown model architecture for feature extraction: {model_name}")
    return feature_extractor


class MomentsAccountant:
    def __init__(self, delta=1e-6):
        self.delta = delta
        self.epsilon = 0.0

    def update(self, sigma_dp, q, steps):
        if sigma_dp == 0:
            self.epsilon = float('inf')
        else:
            self.epsilon = np.sqrt(2 * steps * np.log(1 / self.delta)) * q / sigma_dp


class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint=1.0, output_limits=(4, 32)):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        self._integral = 0
        self._prev_error = 0
        self.history = deque(maxlen=32)

    def __call__(self, process_variable):
        self.history.append(process_variable)
        smoothed_pv = np.mean(self.history)
        error = self.setpoint - smoothed_pv
        self._integral += error
        derivative = error - self._prev_error
        output = self.Kp * error + self.Ki * self._integral + self.Kd * derivative
        self._prev_error = error
        return int(np.clip(output, *self.output_limits))

# (TTA method classes unchanged – omitted for brevity) ---------------------------------
# ... [ KEEP ORIGINAL IMPLEMENTATIONS OF TTAMethod, BN, AdaBN, Tent, RoTTA, ZorroPP ]
# --------------------------------------------------------------------------------------

# --- Main Experiment Runner -----------------------------------------------------------------

def run_experiment(config, dataloader, device):
    results = []

    # NVML initialisation (may fail on CPU-only environments)
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except pynvml.NVMLError:
        logging.warning("pynvml could not be initialised. Power metrics will be unavailable.")
        handle = None

    logging.info(f"Loading model: {config['model']['name']}")
    if 'wav2vec' in config['model']['name']:
        model = Wav2Vec2Model.from_pretrained(config['model']['name']).to(device)
        model.classifier = nn.Linear(model.config.hidden_size, config['dataset']['num_classes']).to(device)
    else:
        model = timm.create_model(
            config['model']['name'],
            pretrained=True,
            num_classes=config['dataset']['num_classes']
        ).to(device)

    method = config.get('method', 'BN')
    logging.info(f"Using TTA method: {method}")
    adaptor_classes = {
        'BN': BN, 'AdaBN': AdaBN, 'Tent': Tent,
        'RoTTA': RoTTA, 'ZorroPP': ZorroPP
    }
    adaptor = adaptor_classes.get(method, BN)(model, config).to(device)

    if method != 'BN':
        adaptor.train()
    else:
        adaptor.eval()

    total_correct = 0
    total_samples = 0

    for i, (inputs, labels) in enumerate(dataloader):
        if i * config['dataloader']['batch_size'] >= config['stream']['frames']:
            break

        inputs, labels = inputs.to(device), labels.to(device)
        latency_ms = 0.0
        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True);
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        outputs = adaptor(inputs)
        if torch.cuda.is_available():
            end_event.record(); torch.cuda.synchronize(); latency_ms = start_event.elapsed_time(end_event)
        power_watts = (pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0) if handle else 0.0

        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()
        accuracy = (total_correct / total_samples) * 100

        if i % 100 == 0:
            logging.info(
                f"Frame {i * config['dataloader']['batch_size']}/{config['stream']['frames']} | "
                f"Acc {accuracy:.2f}% | Lat {latency_ms:.2f} ms | Pwr {power_watts:.2f} W"
            )

        results.append({
            'frame': i * config['dataloader']['batch_size'],
            'accuracy': accuracy,
            'latency_ms': latency_ms,
            'power_watts': power_watts,
        })

    if handle:
        pynvml.nvmlShutdown()

    results_df = pd.DataFrame(results)
    try:
        run_name = make_run_name(config)
    except ValueError:
        logging.error("run_experiment called with a template config – this should never happen.")
        return None

    output_path = os.path.join(config['results_dir'], f"{run_name}.csv")
    results_df.to_csv(output_path, index=False)
    logging.info(f"Results saved to {output_path}")
    return output_path
