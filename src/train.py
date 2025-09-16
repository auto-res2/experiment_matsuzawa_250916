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

# --- Helper Functions & Classes ---

def get_feature_extractor(model, model_name):
    if 'resnet' in model_name or 'mobilenet' in model_name:
        feature_extractor = nn.Sequential(*list(model.children())[:-1])
    elif 'vit' in model_name:
        feature_extractor = model
        # ViT's forward_features or equivalent will be used.
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
            # Simplified DP-SGD to RDP conversion, then to (eps, delta)-DP
            # This is a common approximation for Moments Accountant.
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
        # Use moving average to smooth the process variable
        smoothed_pv = np.mean(self.history)

        error = self.setpoint - smoothed_pv
        self._integral += error
        derivative = error - self._prev_error

        output = self.Kp * error + self.Ki * self._integral + self.Kd * derivative

        self._prev_error = error
        return int(np.clip(output, self.output_limits[0], self.output_limits[1]))


# --- TTA Methods ---

class TTAMethod(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
        self.model_name = config['model']['name']
        self.feature_extractor = get_feature_extractor(self.model, self.model_name)
        if 'resnet' in self.model_name or 'mobilenet' in self.model_name:
            self.classifier = list(model.children())[-1]
        elif 'vit' in self.model_name:
            self.classifier = model.head
        elif 'wav2vec' in self.model_name:
            self.classifier = model.classifier
        else:
            raise ValueError(f"Unknown model architecture for classifier: {self.model_name}")
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        raise NotImplementedError


class BN(TTAMethod):
    def forward(self, x):
        self.model.eval()  # Ensure BN stats are frozen
        return self.model(x)


class AdaBN(TTAMethod):
    def __init__(self, model, config):
        super().__init__(model, config)
        # In AdaBN, we just need to run the model in train mode once per batch
        # to update the running stats.

    def forward(self, x):
        self.model.train()  # This updates the running mean/var of BNs
        return self.model(x)


class Tent(TTAMethod):
    def __init__(self, model, config):
        super().__init__(model, config)
        self.optimizer = optim.SGD(self.model.parameters(), lr=config.get('lr', 1e-4))
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                m.track_running_stats = False

    @staticmethod
    def entropy(p, prob=True):
        if prob:
            p = torch.clamp(p, 1e-6, 1. - 1e-6)
            return -torch.sum(p * torch.log(p), dim=1)
        else:
            p = torch.softmax(p, dim=1)
            return -torch.sum(p * torch.log(p), dim=1)

    def forward(self, x):
        outputs = self.model(x)
        loss = self.entropy(outputs, prob=False).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return outputs


class RoTTA(TTAMethod):
    def __init__(self, model, config):
        super().__init__(model, config)
        self.memory_size = config.get('memory_size', 64)
        self.nu = config.get('nu', 0.001)
        self.gamma = config.get('gamma', 0.9)
        self.lambda_div = config.get('lambda_div', 1.0)
        self.ema_model = timm.create_model(self.model_name, pretrained=True).cuda().eval()
        for param in self.ema_model.parameters():
            param.requires_grad = False

        self.memory_x = deque(maxlen=self.memory_size)
        self.memory_y = deque(maxlen=self.memory_size)

    def forward(self, x):
        # EMA update of model weights
        for param_q, param_k in zip(self.model.parameters(), self.ema_model.parameters()):
            param_k.data = param_k.data * self.gamma + param_q.data * (1. - self.gamma)

        outputs = self.model(x)
        ema_outputs = self.ema_model(x)

        loss_ent = Tent.entropy(outputs, prob=False).mean()

        loss_div = 0
        if len(self.memory_x) > 0:
            mem_x = torch.stack(list(self.memory_x), dim=0)
            mem_y = self.model(mem_x)
            loss_div = -Tent.entropy(mem_y, prob=False).mean()

        loss = loss_ent + self.lambda_div * loss_div
        # Optimization step would go here, assuming we set up an optimizer for `self.model`

        with torch.no_grad():
            pseudo_labels = torch.argmax(ema_outputs, dim=1)
            self.memory_x.extend(x)
            self.memory_y.extend(pseudo_labels)

        return outputs


class ZorroPP(TTAMethod):
    def __init__(self, model, config):
        super().__init__(model, config)
        self.m = config['adaptor_params']['m']
        self.tau = config['adaptor_params']['tau']
        self.d = config['adaptor_params']['d']
        self.sigma_dp = config['adaptor_params'].get('sigma_dp', 0.0)
        self.epsilon_max = config['adaptor_params'].get('epsilon_max', 8.0)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Get feature dimension
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        features = self._extract_features(dummy_input)
        self.C = features.shape[1]

        # Sketch tensors
        self.sketch_mu = torch.zeros(self.C, self.m, device=self.device)
        self.sketch_var = torch.zeros(self.C, self.m, device=self.device)
        self.sketch_skew = torch.zeros(self.C, self.m, device=self.device)
        self.sketch_kurt = torch.zeros(self.C, self.m, device=self.device)
        self.t = 0

        # Hashing for CountSketch
        self.h = torch.randint(0, self.m, (self.C,), device=self.device)
        # Generate random ±1 values for count sketch signs
        self.g = torch.where(torch.rand(self.C, device=self.device) < 0.5,
                             torch.full((self.C,), -1.0, device=self.device),
                             torch.full((self.C,), 1.0, device=self.device))

        self.moments_accountant = MomentsAccountant()
        self.pid_controller = None
        if 'pid_gains' in config['adaptor_params']:
            gains = config['adaptor_params']['pid_gains']
            self.pid_controller = PIDController(gains['Kp'], gains['Ki'], gains['Kd'])
            self.current_rank = 32  # Full rank initially

    def _extract_features(self, x):
        if 'vit' in self.model_name:
            return self.feature_extractor.forward_features(x)[:, 0]
        else:
            x = self.feature_extractor(x)
            return x.view(x.size(0), -1)

    def _update_sketch(self, x):
        self.t += 1
        lambda_t = 1.0 - np.exp(-self.t / self.tau)
        self.sketch_mu *= (1 - lambda_t)
        self.sketch_var *= (1 - lambda_t)
        self.sketch_skew *= (1 - lambda_t)
        self.sketch_kurt *= (1 - lambda_t)

        # Huber-Kalman gain (simplified)
        kappa_t = 1.0 / self.t

        for i in range(x.shape[0]):
            sample = x[i]
            g_s = self.g * sample
            # Update mu sketch
            self.sketch_mu[:, self.h] += kappa_t * g_s[:, None]

        # Higher-order moment updates (simplified for this implementation)
        # A full implementation would update sketches for var, skew, kurtosis here.
        if self.sigma_dp > 0:
            noise = torch.randn_like(self.sketch_mu) * self.sigma_dp
            self.sketch_mu += noise
            self.moments_accountant.update(self.sigma_dp, 1.0, self.t)

    def _get_moments(self):
        mu_hat = torch.mean(self.sketch_mu * self.g[:, None], dim=1)
        # In a full implementation, other moments would be decoded here
        var_hat = torch.ones_like(mu_hat)  # Placeholder
        skew_hat = torch.zeros_like(mu_hat)  # Placeholder
        kurt_hat = torch.ones_like(mu_hat) * 3  # Placeholder (Gaussian kurtosis)
        return mu_hat, var_hat, skew_hat, kurt_hat

    def _chebyshev_normalize(self, x, mu, var, skew, kurt):
        # Center and whiten
        x_norm = (x - mu) / (torch.sqrt(var) + 1e-6)
        # Higher order corrections (simplified)
        if self.d >= 2:
            x_norm = x_norm - 0.5 * skew * (x_norm ** 2 - 1)
        if self.d >= 3:
            x_norm = x_norm / (torch.sqrt(kurt / 8 + 1e-6))
        return x_norm

    def forward(self, x):
        features = self._extract_features(x)

        if self.training:  # Only adapt if in training mode
            self._update_sketch(features)

        mu_hat, var_hat, skew_hat, kurt_hat = self._get_moments()

        norm_features = self._chebyshev_normalize(features, mu_hat, var_hat, skew_hat, kurt_hat)

        # Reshape back to feature map format if needed
        if 'resnet' in self.model_name or 'mobilenet' in self.model_name:
            norm_features = norm_features.view(norm_features.size(0), -1, 1, 1)

        if 'vit' in self.model_name:
            # ViT expects sequence; for simplicity, we directly pass CLS token representation
            return self.model.head(norm_features)
        else:
            return self.classifier(norm_features)

# --- Main Experiment Runner ---

def run_experiment(config, dataloader, device):
    results = []
    # pynvml setup
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except pynvml.NVMLError:
        logging.warning("pynvml could not be initialized. Power metrics will be unavailable.")
        handle = None

    # Model and Adaptor Loading
    logging.info(f"Loading model: {config['model']['name']}")
    if 'wav2vec' in config['model']['name']:
        model = Wav2Vec2Model.from_pretrained(config['model']['name']).to(device)
        # Add a classification head
        model.classifier = nn.Linear(model.config.hidden_size, config['dataset']['num_classes']).to(device)
    else:
        model = timm.create_model(config['model']['name'], pretrained=True, num_classes=config['dataset']['num_classes']).to(device)

    method = config['method']
    logging.info(f"Using TTA method: {method}")
    if method == 'BN':
        adaptor = BN(model, config)
    elif method == 'AdaBN':
        adaptor = AdaBN(model, config)
    elif method == 'Tent':
        adaptor = Tent(model, config)
    elif method == 'RoTTA':
        adaptor = RoTTA(model, config)
    elif method == 'ZorroPP':
        adaptor = ZorroPP(model, config)
    else:
        logging.warning(f"Method {method} not fully implemented. Using BN as default.")
        adaptor = BN(model, config)

    adaptor.to(device)

    # IMPORTANT: Set to train() mode to enable adaptation logic (e.g., sketch updates)
    if method != 'BN':
        adaptor.train()
    else:
        adaptor.eval()

    total_correct = 0
    total_samples = 0
    start_time = time.time()

    for i, (inputs, labels) in enumerate(dataloader):
        if i * config['dataloader']['batch_size'] >= config['stream']['frames']:
            break

        inputs, labels = inputs.to(device), labels.to(device)

        # Performance measurement start (only if CUDA is available)
        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

        outputs = adaptor(inputs)

        if torch.cuda.is_available():
            end_event.record()
            torch.cuda.synchronize()
            latency_ms = start_event.elapsed_time(end_event)
        else:
            latency_ms = (time.time() - start_time) * 1000  # rough estimate

        power_watts = 0
        if handle:
            power_watts = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0

        # Accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()
        accuracy = (total_correct / total_samples) * 100

        # Logging
        if i % 100 == 0:
            logging.info(
                f"Frame {i * config['dataloader']['batch_size']}/{config['stream']['frames']} | "
                f"Accuracy: {accuracy:.2f}% | Latency: {latency_ms:.2f}ms | Power: {power_watts:.2f}W"
            )

        results.append({
            'frame': i * config['dataloader']['batch_size'],
            'accuracy': accuracy,
            'latency_ms': latency_ms,
            'power_watts': power_watts
        })

    if handle:
        pynvml.nvmlShutdown()

    # Save results
    results_df = pd.DataFrame(results)
    run_name = (
        f"{config['exp_name']}_{config['model']['name']}" +
        f"_{config['dataset']['name']}_{config['dataset']['corruption']}" +
        f"_{config['method']}_eta{config['stream']['eta']}_seed{config['seed']}"
    )
    output_path = os.path.join(config['results_dir'], f"{run_name}.csv")
    results_df.to_csv(output_path, index=False)
    logging.info(f"Results saved to {output_path}")

    return output_path
