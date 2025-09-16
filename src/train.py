import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.nn import GATConv
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch_geometric.utils import subgraph as pyg_subgraph
import pynvml

# -----------------------------------------------------------------------------
# Utility: safe torch.load
# -----------------------------------------------------------------------------

def safe_torch_load(path: str, *, map_location=None):
    """Wrapper around torch.load that always uses weights_only=False.
    This restores <2.6 behaviour so we can load arbitrary objects (e.g. PyG Data)."""
    return torch.load(path, map_location=map_location, weights_only=False)

# -----------------------------------------------------------------------------
# Simple fallback sampler (CPU-only, no torch-sparse dependency)                
# -----------------------------------------------------------------------------
class RandomNodeSampler:
    """A very lightweight replacement for GraphSAINTRandomWalkSampler that works
    without torch-sparse.  Each iteration returns an induced sub-graph on a
    random set of *batch_size* nodes.  This is **only** meant for smoke-tests
    and environments where *torch-sparse* cannot be installed.  For real
    experiments, please install torch-sparse so that the official GraphSAINT
    sampler can be used.
    """

    def __init__(self, data, batch_size: int, num_steps: int):
        self.data = data  # must stay on **CPU**
        self.batch_size = max(1, min(batch_size, data.num_nodes))
        self.num_steps = num_steps

    def __iter__(self):
        for _ in range(self.num_steps):
            node_idx = torch.randperm(self.data.num_nodes)[: self.batch_size]
            if hasattr(self.data, "subgraph"):
                sub = self.data.subgraph(node_idx)
            else:
                ei, _ = pyg_subgraph(node_idx, self.data.edge_index, relabel_nodes=True)
                sub = self.data.clone()
                sub.x = self.data.x[node_idx]
                sub.y = self.data.y[node_idx] if self.data.y is not None else None
                for mask_name in ["train_mask", "val_mask", "test_mask"]:
                    m = getattr(self.data, mask_name, None)
                    if m is not None:
                        setattr(sub, mask_name, m[node_idx])
                sub.edge_index = ei
            yield sub

    def __len__(self):
        return self.num_steps

# -----------------------------------------------------------------------------
# Model Definitions
# -----------------------------------------------------------------------------

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
            if predictor is not None and i < len(self.convs) - 1:  # Prune all but last layer
                with torch.no_grad():
                    edge_mask, _ = predictor.get_edge_mask(x, edge_index, layer_idx=i, keep_ratio=keep_ratio)
                pruned_edge_index = edge_index[:, edge_mask]
            else:
                pruned_edge_index = edge_index

            x = F.dropout(x, p=self.dropout, training=self.training)
            if return_attention_weights:
                x, att_w = conv(x, pruned_edge_index, return_attention_weights=True)
                all_attention_weights.append(att_w)
            else:
                x = conv(x, pruned_edge_index)

            if i < len(self.convs) - 1:
                x = F.elu(x)

        if return_attention_weights:
            # Compute full attention for correlation calculation (no pruning)
            with torch.no_grad():
                full_att_weights = []
                temp_x = x.detach()
                for i, conv in enumerate(self.convs):
                    if i < len(self.convs) - 1:
                        _, att = conv(temp_x, original_edge_index, return_attention_weights=True)
                        full_att_weights.append(att)
                        temp_x = conv(temp_x, original_edge_index)
                        temp_x = F.elu(temp_x)
            return F.log_softmax(x, dim=1), full_att_weights
        return F.log_softmax(x, dim=1)

# ---------------- META-LEAP Predictor (unchanged except for import safety) ----
class MetaLEAPPredictor(nn.Module):
    def __init__(self, in_channels, num_layers, heads, meta_ckpt_path, control_variate="ar1", ar1_rho_init=0.9):
        super().__init__()
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.heads = heads
        self.control_variate_type = control_variate

        # A. Meta-initialisation Ψ
        self.psi = safe_torch_load(meta_ckpt_path, map_location="cpu")
        self.psi.eval()

        # B. Online Δ-predictor
        self.delta_w = nn.ParameterList([nn.Parameter(torch.zeros(in_channels * 2)) for _ in range(num_layers - 1)])

        # C. Cross-head factorisation
        self.u = nn.ParameterList([nn.Parameter(torch.randn(in_channels * 2)) for _ in range(num_layers - 1)])
        self.gamma_h = nn.ParameterList([nn.Parameter(torch.ones(heads)) for _ in range(num_layers - 1)])

        self._is_delta_frozen = False

        # D. AR(1) control-variate state
        if self.control_variate_type == "ar1":
            self.ar1_lambda = [ar1_rho_init] * (num_layers - 1)
            self.grad_ewma = [0.0] * (num_layers - 1)
            self.grad_var_ewma = [1.0] * (num_layers - 1)
            self.ar1_rho_hat = [ar1_rho_init] * (num_layers - 1)

    def forward(self, x, edge_index, structural_features, layer_idx):
        with torch.no_grad():
            w0 = self.psi(structural_features)
        w = w0 + self.delta_w[layer_idx]
        row, col = edge_index
        phi_ij = torch.cat([x[row], x[col]], dim=-1)
        base_pred = F.leaky_relu((phi_ij * (self.u[layer_idx] + w)).sum(dim=-1))
        y_hat_h = self.gamma_h[layer_idx].unsqueeze(0) * base_pred.unsqueeze(1)
        return y_hat_h

    @torch.no_grad()
    def get_edge_mask(self, x, edge_index, layer_idx, keep_ratio=0.2):
        row, col = edge_index
        deg = (
            torch.bincount(row, minlength=x.size(0)).to(x.device)
            + torch.bincount(col, minlength=x.size(0)).to(x.device)
        )
        deg_row = torch.log(deg[row].float() + 1e-6)
        deg_col = torch.log(deg[col].float() + 1e-6)
        structural_features = torch.zeros(edge_index.size(1), 4, device=x.device)
        structural_features[:, 0] = deg_row
        structural_features[:, 1] = deg_col
        y_hat_h = self.forward(x, edge_index, structural_features, layer_idx)
        y_hat = y_hat_h.mean(dim=1)
        k = int(keep_ratio * edge_index.size(1))
        if k >= edge_index.size(1):
            return torch.ones(edge_index.size(1), dtype=torch.bool, device=x.device), y_hat
        _, topk = torch.topk(y_hat, k)
        mask = torch.zeros(edge_index.size(1), dtype=torch.bool, device=x.device)
        mask[topk] = True
        return mask, y_hat

    # --- AR(1) helpers (unchanged) -------------------------------------------
    def ar1_update(self, current_grad, layer_idx, alpha=0.01):
        if self.control_variate_type != "ar1":
            return
        g_t = torch.norm(current_grad).item()
        cov = (g_t - self.grad_ewma[layer_idx]) * (
            self.grad_ewma[layer_idx] - self.grad_ewma[layer_idx]
        )
        self.ar1_rho_hat[layer_idx] = (1 - alpha) * self.ar1_rho_hat[layer_idx] + alpha * (
            cov / (self.grad_var_ewma[layer_idx] + 1e-8)
        )
        self.ar1_rho_hat[layer_idx] = np.clip(self.ar1_rho_hat[layer_idx], -1.0, 1.0)
        self.grad_ewma[layer_idx] = (1 - alpha) * self.grad_ewma[layer_idx] + alpha * g_t
        self.grad_var_ewma[layer_idx] = (1 - alpha) * self.grad_var_ewma[layer_idx] + alpha * (
            (g_t - self.grad_ewma[layer_idx]) ** 2
        )
        self.ar1_lambda[layer_idx] = np.clip(self.ar1_rho_hat[layer_idx] ** 2, 0.0, 0.99)

    def apply_control_variate(self, layer_idx):
        if self.control_variate_type == "ar1":
            return 1 - self.ar1_lambda[layer_idx]
        return 1.0

    def freeze_delta(self, frozen: bool):
        self._is_delta_frozen = frozen
        for p in self.delta_w:
            p.requires_grad = not frozen

    def delta_params(self):
        return self.delta_w

# -----------------------------------------------------------------------------
# NVML Power Monitoring helper (unchanged)                                      
# -----------------------------------------------------------------------------
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
            p_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            self.log.append(p_mw / 1000.0)
            time.sleep(self.interval)
        pynvml.nvmlShutdown()

    def get_total_energy_kWh(self):
        if not self.log:
            return 0.0
        joules = sum(self.log) * self.interval
        return joules / 3.6e6

# -----------------------------------------------------------------------------
# DDP helpers
# -----------------------------------------------------------------------------

def setup_ddp(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def get_device(rank):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{rank}")
    return torch.device("cpu")

# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------

def train(rank, world_size, config):
    # DDP setup if requested and multi-GPU present
    if config["training"]["ddp"] and world_size > 1:
        setup_ddp(rank, world_size)

    device = get_device(rank)
    if device.type == "cuda":
        torch.cuda.set_device(device.index)

    torch.manual_seed(config["global_seed"])
    np.random.seed(config["global_seed"])

    output_root = os.path.join(".research", "iteration1")
    os.makedirs(output_root, exist_ok=True)

    for exp_name, exp_cfg in config["experiments"].items():
        if rank == 0:
            print(f"--- Starting Experiment: {exp_name} ---")

        # IMPORTANT: keep full graph on CPU (GraphSAINT sampler requires this)
        data_path = exp_cfg["dataset"]["path"]
        data = safe_torch_load(data_path, map_location="cpu")

        for model_name in exp_cfg["models"]:
            for seed in exp_cfg["seeds"]:
                if rank == 0:
                    print(f"  - Training Model: {model_name} | Seed: {seed}")

                torch.manual_seed(seed)
                np.random.seed(seed)

                mp_cfg = exp_cfg["model_params"]
                model = GraphSAINTGAT(
                    in_channels=data.num_node_features,
                    hidden_channels=mp_cfg["hidden_channels"],
                    out_channels=int(data.y.max().item()) + 1,
                    num_layers=mp_cfg["num_layers"],
                    heads=mp_cfg["heads"],
                    dropout=mp_cfg["dropout"]
                ).to(device)

                if config["training"]["ddp"] and world_size > 1:
                    model = DDP(model, device_ids=[device.index])

                predictor = None
                if model_name in ["leap", "meta-leap"]:
                    psi_path = os.path.join(config["preprocess"]["output_dir"], "psi.pt")
                    predictor = MetaLEAPPredictor(
                        in_channels=mp_cfg["hidden_channels"],
                        num_layers=mp_cfg["num_layers"],
                        heads=mp_cfg["heads"],
                        meta_ckpt_path=psi_path,
                        control_variate=exp_cfg.get(
                            "control_variate", "ar1" if model_name == "meta-leap" else "none"
                        ),
                    ).to(device)

                params = list(model.parameters()) + (list(predictor.parameters()) if predictor else [])
                opt = torch.optim.AdamW(
                    params,
                    lr=exp_cfg["training"]["lr"],
                    weight_decay=exp_cfg["training"]["weight_decay"],
                )
                warm = exp_cfg["training"]["lr_warmup_steps"]
                total = exp_cfg["training"]["epochs"] * exp_cfg["training"]["steps_per_epoch"]
                sched1 = LinearLR(opt, start_factor=0.01, total_iters=warm)
                sched2 = CosineAnnealingLR(opt, T_max=max(1, total - warm))
                scheduler = SequentialLR(opt, [sched1, sched2], milestones=[warm])

                sp = exp_cfg["sampler"]
                # ------------------------------------------------------------------
                # Attempt to build GraphSAINT sampler.  If that fails due to missing
                # torch-sparse, fall back to simple RandomNodeSampler so that the
                # smoke-test can still proceed.
                # ------------------------------------------------------------------
                try:
                    loader = GraphSAINTRandomWalkSampler(
                        data,
                        batch_size=sp["budget"],
                        walk_length=sp["walk_length"],
                        num_steps=exp_cfg["training"]["steps_per_epoch"],
                        sample_coverage=100,
                    )
                except ImportError as e:
                    if "torch-sparse" in str(e):
                        if rank == 0:
                            print(
                                "torch-sparse not available – falling back to RandomNodeSampler (CPU-only)."
                            )
                        # Approximate node batch size so that avg #edges ≈ budget/2
                        est_nodes = max(1, int(sp["budget"] / max(1, (data.num_edges / data.num_nodes)) // 2))
                        loader = RandomNodeSampler(
                            data,
                            batch_size=est_nodes,
                            num_steps=exp_cfg["training"]["steps_per_epoch"],
                        )
                    else:
                        raise

                run_dir = os.path.join(output_root, f"{exp_name}_{model_name}_seed{seed}")
                os.makedirs(run_dir, exist_ok=True)
                log = []

                pwr_monitor = None
                if (
                    rank == 0
                    and config["training"]["monitor_power"]
                    and device.type == "cuda"
                ):
                    pwr_monitor = PowerMonitor(device.index)
                    pwr_monitor.start()

                for epoch in range(exp_cfg["training"]["epochs"]):
                    model.train()
                    if predictor:
                        predictor.train()
                    st = time.time()
                    total_loss = 0.0

                    # Special adaptation logic for transfer experiment
                    if exp_name == "exp2_transfer" and predictor is not None:
                        if epoch == 0:
                            predictor.freeze_delta(True)
                        elif epoch == 1 and predictor._is_delta_frozen:
                            predictor.freeze_delta(False)
                            opt.add_param_group(
                                {
                                    "params": predictor.delta_params(),
                                    "lr": exp_cfg["training"]["lr"] * 5.0,
                                }
                            )

                    for subgraph in loader:
                        subgraph = subgraph.to(device)
                        opt.zero_grad()
                        out = model(subgraph.x, subgraph.edge_index, predictor=predictor)
                        loss = F.nll_loss(
                            out[subgraph.train_mask], subgraph.y[subgraph.train_mask]
                        )
                        loss.backward()

                        if predictor and model_name == "meta-leap":
                            for idx, p in enumerate(predictor.delta_w):
                                if p.grad is not None:
                                    predictor.ar1_update(p.grad, idx)
                                    p.grad *= predictor.apply_control_variate(idx)

                        opt.step()
                        scheduler.step()
                        total_loss += loss.item()

                    epoch_time = time.time() - st
                    avg_loss = total_loss / len(loader)

                    if rank == 0:
                        val_acc, test_acc, rho = evaluate(model, predictor, data, device)
                        mem = (
                            torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                            if device.type == "cuda"
                            else 0.0
                        )
                        entry = {
                            "epoch": epoch,
                            "loss": avg_loss,
                            "val_acc": val_acc,
                            "test_acc": test_acc,
                            "epoch_time_s": epoch_time,
                            "gpu_mem_gb": mem,
                            "predictor_corr_rho": rho,
                            "lambda_t_avg": float(
                                np.mean(getattr(predictor, "ar1_lambda", [0.0]))
                            )
                            if predictor and model_name == "meta-leap"
                            else 0.0,
                        }
                        log.append(entry)
                        print(
                            f"Epoch {epoch:02d} | Loss {avg_loss:.4f} | Val {val_acc:.4f} | "
                            f"Test {test_acc:.4f} | Time {epoch_time:.2f}s | Rho {rho:.3f}"
                        )
                        # Save best-by-val
                        if val_acc >= max([l["val_acc"] for l in log]):
                            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pt"))

                if rank == 0:
                    energy = 0.0
                    if pwr_monitor:
                        pwr_monitor.stop()
                        energy = pwr_monitor.get_total_energy_kWh()
                    with open(os.path.join(run_dir, "train_log.json"), "w") as f:
                        json.dump({"log": log, "total_energy_kWh": energy}, f, indent=4)

    if config["training"]["ddp"] and world_size > 1:
        cleanup_ddp()

# -----------------------------------------------------------------------------
# Evaluation helper (sub-sampled)                                              
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, predictor, data, device, subset_nodes=10000):
    model.eval()
    if predictor:
        predictor.eval()

    subset_nodes = min(subset_nodes, data.num_nodes)  # safety for tiny graphs

    node_idx = torch.randperm(data.num_nodes)[:subset_nodes]

    if hasattr(data, "subgraph"):
        sub = data.subgraph(node_idx)
    else:
        ei, _ = pyg_subgraph(node_idx, data.edge_index, relabel_nodes=True)
        sub = data.clone()
        sub.x = data.x[node_idx]
        sub.y = data.y[node_idx] if data.y is not None else None
        for mask_name in ["train_mask", "val_mask", "test_mask"]:
            m = getattr(data, mask_name, None)
            if m is not None:
                setattr(sub, mask_name, m[node_idx])
        sub.edge_index = ei

    sub = sub.to(device)
    out = model(sub.x, sub.edge_index)
    pred = out.argmax(dim=-1)

    val_acc = (
        (pred[sub.val_mask] == sub.y[sub.val_mask]).float().mean().item()
        if sub.val_mask.sum() > 0
        else 0.0
    )
    test_acc = (
        (pred[sub.test_mask] == sub.y[sub.test_mask]).float().mean().item()
        if sub.test_mask.sum() > 0
        else 0.0
    )

    rho = 0.0
    if predictor is not None:
        base_model = model.module if isinstance(model, DDP) else model
        _, att_list = base_model(sub.x, sub.edge_index, return_attention_weights=True)
        true_att = att_list[0][1]  # (E, H)
        true_avg = true_att.mean(dim=1)
        _, pred_scores = predictor.get_edge_mask(sub.x, sub.edge_index, layer_idx=0, keep_ratio=1.0)
        vx = pred_scores - pred_scores.mean()
        vy = true_avg - true_avg.mean()
        rho = (vx * vy).sum() / (
            torch.sqrt((vx ** 2).sum()) * torch.sqrt((vy ** 2).sum()) + 1e-8
        )
        rho = rho.item()

    return val_acc, test_acc, rho

# -----------------------------------------------------------------------------
# Entry point called from src/main.py
# -----------------------------------------------------------------------------

def main(config):
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if config["training"]["ddp"] and world_size > 1:
        import torch.multiprocessing as mp

        mp.spawn(train, args=(world_size, config), nprocs=world_size, join=True)
    else:
        train(0, 1, config)