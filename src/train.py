import os
import logging
import torch
import torch.nn as nn
import timm
from transformers import Wav2Vec2Model
from collections import deque
import numpy as np


class ZORROppAdaptor(nn.Module):
    """ Implements the ZORRO++ adaptation logic. """
    def __init__(self, model: nn.Module, config: dict):
        super().__init__()
        self.model = model
        self.config = config
        self.adaptor_params = config.get('adaptor', {}).get('params', {})

        self.feature_dim = self.get_feature_dim(model)
        self.m = self.adaptor_params.get('m', 16)
        self.tau = self.adaptor_params.get('tau', 500)
        self.chebyshev_order = self.adaptor_params.get('chebyshev_order', 2)
        self.dp_sigma = self.adaptor_params.get('dp_sigma', 0.0)
        self.energy_aware = self.adaptor_params.get('energy_aware', False)

        # Sketching matrices for 4 moments
        self.register_buffer('proj_h1', torch.randn(self.m, self.feature_dim))
        self.register_buffer('proj_s1', torch.randint(0, 2, (self.m, self.feature_dim)) * 2 - 1)
        self.register_buffer('proj_h2', torch.randn(self.m, self.feature_dim))
        self.register_buffer('proj_s2', torch.randint(0, 2, (self.m, self.feature_dim)) * 2 - 1)
        self.register_buffer('proj_h3', torch.randn(self.m, self.feature_dim))
        self.register_buffer('proj_s3', torch.randint(0, 2, (self.m, self.feature_dim)) * 2 - 1)
        self.register_buffer('proj_h4', torch.randn(self.m, self.feature_dim))
        self.register_buffer('proj_s4', torch.randint(0, 2, (self.m, self.feature_dim)) * 2 - 1)

        # Moment sketches
        self.register_buffer('mu_sketch', torch.zeros(self.m))
        self.register_buffer('var_sketch', torch.zeros(self.m))
        self.register_buffer('skew_sketch', torch.zeros(self.m))
        self.register_buffer('kurt_sketch', torch.zeros(self.m))
        
        self.register_buffer('source_mu', torch.zeros(self.feature_dim))
        self.register_buffer('source_var', torch.ones(self.feature_dim))

        self.t = 0

        if self.energy_aware:
            pid_params = self.adaptor_params.get('pid_params', {'kp': 0.4, 'ki': 0.05, 'kd': 0.01})
            self.pid = PIDController(pid_params['kp'], pid_params['ki'], pid_params['kd'])
            self.rank = self.m
            self.chebyshev_order_dyn = self.chebyshev_order

        if self.dp_sigma > 0:
            from opacus.accountants import RDPAccountant
            self.privacy_accountant = RDPAccountant()

    def get_feature_dim(self, model):
        # A simple heuristic to get feature dimension
        if hasattr(model, 'num_features'):
            return model.num_features
        if hasattr(model, 'classifier'):
            if isinstance(model.classifier, nn.Linear):
                return model.classifier.in_features
            if isinstance(model.classifier, nn.Sequential) and isinstance(model.classifier[-1], nn.Linear):
                return model.classifier[-1].in_features
        if hasattr(model, 'fc'):
            return model.fc.in_features
        # Fallback for models like wav2vec
        if 'wav2vec' in model.name_or_path:
            return model.config.hidden_size
        
        # Last resort: run a dummy forward pass
        try:
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224).to(next(model.parameters()).device)
                features = model.forward_features(dummy_input)
                return features.shape[-1]
        except Exception:
            raise ValueError("Could not determine feature dimension for the model.")

    def update(self, features, power=None, latency=None):
        self.t += 1
        features = features.view(features.size(0), -1).detach()
        
        # Energy-aware scheduling
        if self.energy_aware and power is not None and latency is not None:
            l_max = self.adaptor_params.get('latency_budget_ms', 2.0)
            p_max = self.adaptor_params.get('power_budget_w', 2.0)
            error = max(latency / l_max, power / p_max) - 1.0
            self.rank = self.pid(error)
            self.chebyshev_order_dyn = max(1, int(np.ceil(self.rank / (self.m / 3))))

        # Polynomial forget factor
        forget_factor = 1.0 - np.exp(-self.t / self.tau)

        # Update sketches
        self._update_sketch(self.mu_sketch, features, self.proj_h1, self.proj_s1, forget_factor)
        self._update_sketch(self.var_sketch, torch.pow(features - self.get_moments()[0], 2), self.proj_h2, self.proj_s2, forget_factor)
        self._update_sketch(self.skew_sketch, torch.pow(features - self.get_moments()[0], 3), self.proj_h3, self.proj_s3, forget_factor)
        self._update_sketch(self.kurt_sketch, torch.pow(features - self.get_moments()[0], 4), self.proj_h4, self.proj_s4, forget_factor)

        if self.dp_sigma > 0:
            self.privacy_accountant.step(noise_multiplier=self.dp_sigma, sample_rate=features.size(0)/50000.0) # Assume 50k dataset size for sample rate

    def _update_sketch(self, sketch, tensor, proj_h, proj_s, forget_factor):
        # Simplified CountSketch update
        projected = tensor @ (proj_h.T * proj_s.T)
        update = torch.mean(projected, dim=0)
        
        if self.dp_sigma > 0:
            update += torch.randn_like(update) * self.dp_sigma

        sketch.mul_(1-forget_factor).add_(update, alpha=forget_factor)

    def get_moments(self):
        mu = self.mu_sketch @ (self.proj_h1 * self.proj_s1)
        var = self.var_sketch @ (self.proj_h2 * self.proj_s2)
        var = torch.clamp(var, min=1e-6)
        std = torch.sqrt(var)
        skew = (self.skew_sketch @ (self.proj_h3 * self.proj_s3)) / torch.pow(std, 3)
        kurt = (self.kurt_sketch @ (self.proj_h4 * self.proj_s4)) / torch.pow(std, 4)
        return mu, var, skew, kurt

    def transform(self, features):
        mu, var, skew, kurt = self.get_moments()
        
        # Center and Whiten
        x = (features - mu) / torch.sqrt(var + 1e-6)
        
        # De-skew and De-kurtose using Chebyshev polynomials
        order = self.chebyshev_order_dyn if self.energy_aware else self.chebyshev_order
        if order >= 2:
            # P2(x) = 2x^2 - 1, we use a simplified version for de-skewing
            skew_factor = torch.reciprocal(torch.sqrt(torch.abs(skew) + 1e-6))
            x = torch.sign(x) * (torch.pow(torch.abs(x), skew_factor))
        if order >= 3:
            # P3(x) = 4x^3 - 3x, simplified for de-kurtosing
            kurt_factor = torch.reciprocal(torch.sqrt(torch.abs(kurt) + 1e-6))
            x = torch.sign(x) * (torch.pow(torch.abs(x), kurt_factor))
        
        # Recenter to source stats
        transformed_features = x * torch.sqrt(self.source_var + 1e-6) + self.source_mu
        return transformed_features

    def forward(self, x):
        if 'vit' in self.model.default_cfg['architecture']:
            features = self.model.forward_features(x)
            if isinstance(features, list):
                features = features[-1]
            if features.ndim == 3:
                 features = features[:, 0] # CLS token
        else:
            features = self.model.forward_features(x)
            if features.ndim > 2:
                features = self.model.forward_head(features, pre_logits=True)

        self.update(features.detach())
        transformed_features = self.transform(features)
        return self.model.head.fc(transformed_features)

    def bound_mu(self):
        return np.sqrt(np.log(self.feature_dim) / self.t) if self.t > 0 else float('inf')

    def get_privacy_loss(self, delta=1e-5):
        if self.dp_sigma > 0:
            return self.privacy_accountant.get_epsilon(delta)
        return 0.0

class PIDController:
    def __init__(self, Kp, Ki, Kd, output_limits=(4, 32), window=32):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.output_limits = output_limits
        self._integral = 0
        self._prev_error = 0
        self.history = deque(maxlen=window)
        self.setpoint = 0.0 # Target error is 0

    def __call__(self, error):
        self.history.append(error)
        smoothed_error = np.mean(list(self.history))
        self._integral += smoothed_error
        derivative = smoothed_error - self._prev_error
        
        output_adjustment = self.Kp * smoothed_error + self.Ki * self._integral + self.Kd * derivative
        self._prev_error = smoothed_error
        
        # PID adjusts current rank. We assume a starting rank.
        # A negative output means we need to reduce complexity.
        current_output = self.output_limits[1] # Start with max
        new_output = current_output - output_adjustment 
        return int(np.clip(new_output, self.output_limits[0], self.output_limits[1]))

def load_model_and_adaptor(config: dict, device: torch.device):
    model_name = config['model']['name']
    num_classes = config['dataset']['num_classes']
    method = config['method']
    logging.info(f"Loading model: {model_name}")

    if 'wav2vec' in model_name:
        model = Wav2Vec2Model.from_pretrained(model_name).to(device)
        model.classifier = nn.Linear(model.config.hidden_size, num_classes).to(device)
        # A bit of a hack to make it compatible with timm's structure
        model.forward_features = lambda x: model(x).last_hidden_state.mean(dim=1)
        model.head = nn.Identity()
        model.num_features = model.config.hidden_size
        model.name_or_path = model_name
    else:
        model = timm.create_model(model_name, pretrained=True, num_classes=num_classes).to(device)

    # Freeze all model weights
    for param in model.parameters():
        param.requires_grad = False
    
    # All baselines are simplified for this experiment and only ZORRO++ is fully implemented
    if method == 'ZORROpp':
        adaptor = ZORROppAdaptor(model, config)
    else: # Fallback to a simple BatchNorm based adaptor for baselines
        model.train() # BN needs to be in training mode to update stats
        adaptor = model 

    return model.to(device), adaptor.to(device)