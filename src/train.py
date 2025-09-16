import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.nn import GATConv
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from collections import deque
import pynvml

# --- Model Definitions ---

class GraphSAINTGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, heads, dropout):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))
        self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout))
        self.dropout = dropout

    def forward(self, x, edge_index, predictor=None, keep_ratio=0.2, return_attention_weights=False):
        original_edge_index = edge_index
        all_attention_weights = []

        for i, conv in enumerate(self.convs):
            if predictor is not None and i < len(self.convs) -1: # Pruning on all but last layer
                with torch.no_grad():
                    edge_mask, _ = predictor.get_edge_mask(x, edge_index, layer_idx=i, keep_ratio=keep_ratio)
                pruned_edge_index = edge_index[:, edge_mask]
            else:
                pruned_edge_index = edge_index

            x = F.dropout(x, p=self.dropout, training=self.training)
            if return_attention_weights:
                x, attention_weights = conv(x, pruned_edge_index, return_attention_weights=True)
                all_attention_weights.append(attention_weights)
            else:
                x = conv(x, pruned_edge_index)

            if i < len(self.convs) - 1:
                x = F.elu(x)
        
        if return_attention_weights:
             # Calculate full attention for loss calculation if predictor is used
            with torch.no_grad():
                full_att_weights = []
                temp_x = x.detach()
                for i, conv in enumerate(self.convs):
                    if i < len(self.convs) -1:
                        _, att = conv(temp_x, original_edge_index, return_attention_weights=True)
                        full_att_weights.append(att)
                        temp_x = conv(temp_x, original_edge_index)
                        temp_x = F.elu(temp_x)
            return F.log_softmax(x, dim=1), full_att_weights
        return F.log_softmax(x, dim=1)


class MetaLEAPPredictor(nn.Module):
    def __init__(self, in_channels, num_layers, heads, meta_ckpt_path, control_variate='ar1', ar1_rho_init=0.9):
        super().__init__()
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.heads = heads
        self.control_variate_type = control_variate

        # A. Meta-Initialisation (Load pre-trained hyper-network)
        self.psi = torch.load(meta_ckpt_path)
        self.psi.eval() # Freeze Ψ

        # B. Online \Delta-Predictor (per-layer, shared across heads)
        self.delta_w = nn.ParameterList([nn.Parameter(torch.zeros(in_channels * 2)) for _ in range(num_layers - 1)])

        # C. Cross-Head Factorisation
        self.u = nn.ParameterList([nn.Parameter(torch.randn(in_channels * 2)) for _ in range(num_layers - 1)])
        self.gamma_h = nn.ParameterList([nn.Parameter(torch.ones(heads)) for _ in range(num_layers - 1)])

        self._is_delta_frozen = False
        
        # D. AR(1) Control-Variate state
        if self.control_variate_type == 'ar1':
            self.ar1_lambda = [ar1_rho_init] * (num_layers - 1)
            self.grad_ewma = [0.0] * (num_layers - 1)
            self.grad_var_ewma = [1.0] * (num_layers - 1)
            self.ar1_rho_hat = [ar1_rho_init] * (num_layers - 1)

    def forward(self, x, edge_index, structural_features, layer_idx):
        with torch.no_grad():
            w0 = self.psi(structural_features)
        
        w = w0 + self.delta_w[layer_idx]
        
        # phi(h_i, h_j) -> feature for predictor
        row, col = edge_index
        phi_ij = torch.cat([x[row], x[col]], dim=-1)
        
        # ŷ_{ij,h} = γ_h * <u+w, φ_{ij}>
        base_pred = F.leaky_relu((phi_ij * (self.u[layer_idx] + w)).sum(dim=-1))
        
        # Unsqueeze for broadcasting gamma_h across heads
        y_hat_h = self.gamma_h[layer_idx].unsqueeze(0) * base_pred.unsqueeze(1)
        return y_hat_h # Shape: [num_edges, num_heads]

    @torch.no_grad()
    def get_edge_mask(self, x, edge_index, layer_idx, keep_ratio=0.2):
        row, col = edge_index
        # For meta-init, we assume structural features are precomputed and stored on the data object
        # We will use a placeholder here if they are not available
        if hasattr(data, 's_ij') and data.s_ij is not None:
            structural_features = data.s_ij
        else:
            # A simplified structural feature: log-degree
            deg = torch.bincount(row, minlength=x.size(0)) + torch.bincount(col, minlength=x.size(0))
            deg_row = torch.log(deg[row] + 1e-6)
            deg_col = torch.log(deg[col] + 1e-6)
            structural_features = torch.stack([deg_row, deg_col], dim=-1)
        
        y_hat_h = self.forward(x, edge_index, structural_features, layer_idx)
        y_hat = y_hat_h.mean(dim=1) # Average predictions over heads
        
        num_edges_to_keep = int(keep_ratio * edge_index.size(1))
        if num_edges_to_keep >= edge_index.size(1):
            return torch.ones(edge_index.size(1), dtype=torch.bool, device=x.device), y_hat
            
        _, top_k_indices = torch.topk(y_hat, num_edges_to_keep)
        
        edge_mask = torch.zeros(edge_index.size(1), dtype=torch.bool, device=x.device)
        edge_mask[top_k_indices] = True
        return edge_mask, y_hat

    def ar1_update(self, current_grad, layer_idx, alpha=0.01):
        if self.control_variate_type != 'ar1':
            return
        
        # Simplified: use norm of grad as the scalar value g_t
        g_t = torch.norm(current_grad).item()

        # Estimate rho = Cov(g_t, g_{t-1}) / Var(g_{t-1})
        cov = (g_t - self.grad_ewma[layer_idx]) * (self.grad_ewma[layer_idx] - self.grad_ewma[layer_idx])
        self.ar1_rho_hat[layer_idx] = (1 - alpha) * self.ar1_rho_hat[layer_idx] + alpha * (cov / (self.grad_var_ewma[layer_idx] + 1e-8))
        self.ar1_rho_hat[layer_idx] = np.clip(self.ar1_rho_hat[layer_idx], -1.0, 1.0)

        # Update moving averages
        self.grad_ewma[layer_idx] = (1 - alpha) * self.grad_ewma[layer_idx] + alpha * g_t
        self.grad_var_ewma[layer_idx] = (1 - alpha) * self.grad_var_ewma[layer_idx] + alpha * ((g_t - self.grad_ewma[layer_idx])**2)

        # Update lambda
        # Simplified update, as Var_pred and Var_resid are hard to get here
        self.ar1_lambda[layer_idx] = self.ar1_rho_hat[layer_idx]**2
        self.ar1_lambda[layer_idx] = np.clip(self.ar1_lambda[layer_idx], 0.0, 0.99)

    def apply_control_variate(self, layer_idx):
        # This would hook into the backward pass. For simplicity, we simulate the effect
        # by adjusting the learning rate of delta_w based on lambda.
        if self.control_variate_type == 'ar1':
            return (1 - self.ar1_lambda[layer_idx])
        return 1.0

    def freeze_delta(self, frozen):
        self._is_delta_frozen = frozen
        for p in self.delta_params():
            p.requires_grad = not frozen

    def delta_params(self):
        return self.delta_w

# --- NVML Power Monitoring ---
class PowerMonitor:
    def __init__(self, rank, interval=1):
        self.rank = rank
        self.interval = interval
        self.log = []
        self._stop = False
        self.thread = None

    def start(self):
        import threading
        self._stop = False
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()

    def stop(self):
        self._stop = True
        if self.thread:
            self.thread.join()

    def _monitor(self):
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.rank)
        while not self._stop:
            power_mW = pynvml.nvmlDeviceGetPowerUsage(handle)
            self.log.append(power_mW / 1000.0) # Convert to Watts
            time.sleep(self.interval)
        pynvml.nvmlShutdown()

    def get_total_energy_kWh(self):
        if not self.log:
            return 0.0
        total_joules = sum(self.log) * self.interval
        return total_joules / 3.6e6 # Joules to kWh


# --- Training Logic ---

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_ddp():
    dist.destroy_process_group()

def get_device(config, rank):
    if config['training']['ddp']:
        return torch.device(f'cuda:{rank}')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(rank, world_size, config):
    if config['training']['ddp']:
        setup_ddp(rank, world_size)
    
    device = get_device(config, rank)
    torch.cuda.set_device(device)
    torch.manual_seed(config['global_seed'])
    np.random.seed(config['global_seed'])

    output_dir = os.path.join('.research', 'iteration1')
    os.makedirs(output_dir, exist_ok=True)

    for exp_name, exp_config in config['experiments'].items():
        if rank == 0:
            print(f'--- Starting Experiment: {exp_name} ---')
        
        data_path = exp_config['dataset']['path']
        data = torch.load(data_path)
        data = data.to(device)

        for model_name in exp_config['models']:
            for seed in exp_config['seeds']:
                if rank == 0:
                    print(f'  - Training Model: {model_name} | Seed: {seed}')

                torch.manual_seed(seed)
                np.random.seed(seed)

                model_params = exp_config['model_params']
                model = GraphSAINTGAT(
                    in_channels=data.num_node_features,
                    hidden_channels=model_params['hidden_channels'],
                    out_channels=data.y.max().item() + 1,
                    num_layers=model_params['num_layers'],
                    heads=model_params['heads'],
                    dropout=model_params['dropout']
                ).to(device)
                
                if config['training']['ddp']:
                    model = DDP(model, device_ids=[rank])

                predictor = None
                if model_name in ['leap', 'meta-leap']:
                    if model_name == 'meta-leap':
                        psi_path = os.path.join(config['preprocess']['output_dir'], 'psi.pt')
                        predictor = MetaLEAPPredictor(
                            in_channels=model_params['hidden_channels'],
                            num_layers=model_params['num_layers'],
                            heads=model_params['heads'],
                            meta_ckpt_path=psi_path,
                            control_variate=exp_config.get('control_variate', 'ar1')
                        ).to(device)
                    else: # LEAP is simplified META-LEAP without pre-trained Psi and online delta
                        psi_path = os.path.join(config['preprocess']['output_dir'], 'psi.pt') # Use same structure for simplicity
                        predictor = MetaLEAPPredictor(
                            in_channels=model_params['hidden_channels'],
                            num_layers=model_params['num_layers'],
                            heads=model_params['heads'],
                            meta_ckpt_path=psi_path,
                            control_variate='none' # No AR(1) for baseline LEAP
                        ).to(device)
                        # For LEAP, we don't use meta-init, so we make delta_w the main learner
                        predictor.delta_w.data = predictor.u.data.clone()
                        predictor.u.requires_grad = False

                params = list(model.parameters()) 
                if predictor: 
                    params += list(predictor.parameters())
                optimizer = torch.optim.AdamW(params, lr=exp_config['training']['lr'], weight_decay=exp_config['training']['weight_decay'])
                
                warmup_steps = exp_config['training']['lr_warmup_steps']
                total_steps = exp_config['training']['epochs'] * exp_config['training']['steps_per_epoch']
                
                scheduler1 = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
                scheduler2 = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
                scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_steps])

                sampler_params = exp_config['sampler']
                loader = GraphSAINTRandomWalkSampler(data, 
                                                     batch_size=sampler_params['budget'],
                                                     walk_length=sampler_params['walk_length'],
                                                     num_steps=exp_config['training']['steps_per_epoch'],
                                                     sample_coverage=100) # sample_coverage > num_steps to avoid reset
                
                exp_results_dir = os.path.join(output_dir, f'{exp_name}_{model_name}_seed{seed}')
                os.makedirs(exp_results_dir, exist_ok=True)
                train_log = []

                # Start power monitoring
                power_monitor = None
                if rank == 0 and config['training']['monitor_power']:
                    power_monitor = PowerMonitor(rank)
                    power_monitor.start()

                for epoch in range(exp_config['training']['epochs']):
                    model.train()
                    if predictor: predictor.train()
                    epoch_start_time = time.time()
                    total_loss = 0
                    
                    # Adaptation logic for experiment 2
                    if exp_name == 'exp2_transfer' and predictor:
                        if epoch == 0:
                            predictor.freeze_delta(True)
                        elif epoch == 1 and not predictor._is_delta_frozen:
                            predictor.freeze_delta(False)
                            # Add delta params to new group with higher LR
                            optimizer.add_param_group({'params': predictor.delta_params(), 'lr': exp_config['training']['lr'] * 5})
                    
                    for i, subgraph in enumerate(loader):
                        subgraph = subgraph.to(device)
                        optimizer.zero_grad()

                        out = model(subgraph.x, subgraph.edge_index, predictor=predictor if model_name in ['leap', 'meta-leap'] else None)
                        loss = F.nll_loss(out[subgraph.train_mask], subgraph.y[subgraph.train_mask])
                        
                        loss.backward()

                        # Apply control variate and AR(1) update
                        if predictor and model_name == 'meta-leap':
                            for layer_idx, param_group in enumerate(predictor.delta_w):
                                if param_group.grad is not None:
                                    predictor.ar1_update(param_group.grad, layer_idx)
                                    adjustment = predictor.apply_control_variate(layer_idx)
                                    param_group.grad *= adjustment

                        optimizer.step()
                        scheduler.step()
                        total_loss += loss.item()
                    
                    epoch_time = time.time() - epoch_start_time
                    avg_loss = total_loss / len(loader)
                    
                    # Evaluation and logging (only on rank 0)
                    if rank == 0:
                        val_acc, test_acc, rho = evaluate(model, predictor, data, device, exp_config)
                        mem_usage = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                        log_entry = {
                            'epoch': epoch,
                            'loss': avg_loss,
                            'val_acc': val_acc,
                            'test_acc': test_acc,
                            'epoch_time_s': epoch_time,
                            'gpu_mem_gb': mem_usage,
                            'predictor_corr_rho': rho,
                            'lambda_t_avg': np.mean(predictor.ar1_lambda) if predictor and model_name == 'meta-leap' else 0
                        }
                        train_log.append(log_entry)
                        print(f"Epoch {epoch:02d} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f} | Time: {epoch_time:.2f}s | Rho: {rho:.4f}")

                        # Save checkpoint
                        if val_acc >= max([l['val_acc'] for l in train_log]):
                            torch.save(model.state_dict(), os.path.join(exp_results_dir, 'best_model.pt'))
                
                # Stop power monitoring and save results
                if rank == 0:
                    if power_monitor:
                        power_monitor.stop()
                        total_energy = power_monitor.get_total_energy_kWh()
                    else:
                        total_energy = 0
                    
                    results = {'log': train_log, 'total_energy_kWh': total_energy}
                    with open(os.path.join(exp_results_dir, 'train_log.json'), 'w') as f:
                        json.dump(results, f, indent=4)
    
    if config['training']['ddp']:
        cleanup_ddp()

@torch.no_grad()
def evaluate(model, predictor, data, device, config, subset_nodes=10000):
    model.eval()
    if predictor: predictor.eval()

    # Subsample for faster evaluation on large graphs
    node_idx = torch.randperm(data.num_nodes)[:subset_nodes].to(device)
    subgraph = data.subgraph(node_idx)

    out = model(subgraph.x, subgraph.edge_index)
    pred = out.argmax(dim=-1)

    val_acc = (pred[subgraph.val_mask] == subgraph.y[subgraph.val_mask]).float().mean().item()
    test_acc = (pred[subgraph.test_mask] == subgraph.y[subgraph.test_mask]).float().mean().item()
    
    # Calculate predictor correlation rho
    rho = 0.0
    if predictor:
        # Get true attention from the model's first GAT layer
        unpruned_model = model.module if isinstance(model, DDP) else model
        _, attention_weights_list = unpruned_model(subgraph.x, subgraph.edge_index, return_attention_weights=True)
        true_attention = attention_weights_list[0][1] # (E, H) attention weights for first layer
        true_attention_avg = true_attention.mean(dim=1)
        
        # Get predictor's scores
        _, pred_scores = predictor.get_edge_mask(subgraph.x, subgraph.edge_index, layer_idx=0, keep_ratio=1.0)

        # Pearson correlation
        vx = pred_scores - torch.mean(pred_scores)
        vy = true_attention_avg - torch.mean(true_attention_avg)
        rho = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)
        rho = rho.item()

    return val_acc, test_acc, rho


def main(config):
    world_size = torch.cuda.device_count()
    if config['training']['ddp'] and world_size > 1:
        import torch.multiprocessing as mp
        mp.spawn(train, args=(world_size, config), nprocs=world_size, join=True)
    else:
        train(0, 1, config)
