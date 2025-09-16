import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel, wilcoxon
import torch
from tqdm import tqdm
import time
import logging
from fvcore.nn import FlopCountAnalysis

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    logging.warning("pynvml not found. Power metrics will not be available.")

from .train import load_model_and_adaptor

# -----------------------------------------------------------------------------
# NOTE:  All save paths ***must*** point to .research/iteration14 according to
#        the mandatory rules.
# -----------------------------------------------------------------------------

IMAGES_DIR = ".research/iteration14/images"
JSON_DIR   = ".research/iteration14"
os.makedirs(IMAGES_DIR, exist_ok=True)


def run_single_experiment(config: dict, dataloader: torch.utils.data.DataLoader, device: torch.device) -> pd.DataFrame:
    """Run a single experiment configuration and return a DataFrame of per-frame metrics."""
    model, adaptor = load_model_and_adaptor(config, device)

    if NVML_AVAILABLE and torch.cuda.is_available():
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    metrics = []
    total_correct = 0
    total_samples = 0
    frame_idx = 0
    last_moments = None

    pbar = tqdm(total=config['stream']['frames'], desc=f"Running {config['method']}")

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        if frame_idx >= config['stream']['frames']:
            break

        inputs, labels = inputs.to(device), labels.to(device)
        batch_size = inputs.size(0)

        start_time = time.perf_counter()
        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            outputs = adaptor(inputs)

        if torch.cuda.is_available():
            end_event.record()
            torch.cuda.synchronize()
            latency_ms = start_event.elapsed_time(end_event)
        else:
            latency_ms = (time.perf_counter() - start_time) * 1000

        power_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0 if (NVML_AVAILABLE and torch.cuda.is_available()) else 0.0

        _, predicted = torch.max(outputs.data, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += batch_size
        online_acc = (total_correct / total_samples) * 100

        # Collect detailed metrics from ZORROpp
        is_zorro = isinstance(adaptor, torch.nn.Module) and hasattr(adaptor, 'bound_mu')
        if is_zorro:
            bound_mu = adaptor.bound_mu()
            privacy_eps = adaptor.get_privacy_loss()
            current_mu, _, _, _ = adaptor.get_moments()
            if batch_idx > 0 and (batch_idx % 16 == 0):  # drift index every 256 frames (16*16)
                drift_index = torch.linalg.norm(current_mu - last_moments, ord=1).item() if last_moments is not None else 0.0
            else:
                drift_index = np.nan
            if batch_idx % 16 == 0:
                last_moments = current_mu.clone()
        else:
            bound_mu, privacy_eps, drift_index = np.nan, np.nan, np.nan

        metrics.append({
            'frame': frame_idx,
            'online_acc': online_acc,
            'latency_ms': latency_ms / batch_size,
            'power_w': power_w,
            'memory_mb': torch.cuda.max_memory_reserved(device) / 1e6 if torch.cuda.is_available() else 0.0,
            'bound_mu': bound_mu,
            'privacy_eps': privacy_eps,
            'drift_index': drift_index
        })

        frame_idx += batch_size
        pbar.update(batch_size)
        pbar.set_postfix({'Acc': f'{online_acc:.2f}%'})

    pbar.close()
    if NVML_AVAILABLE and torch.cuda.is_available():
        pynvml.nvmlShutdown()

    results_df = pd.DataFrame(metrics)

    # Compute FLOPs for one sample
    try:
        example_input = next(iter(dataloader))[0][:1].to(device)
        flops = FlopCountAnalysis(model, example_input).total()
        results_df['flops_g'] = flops / 1e9
    except Exception as e:
        logging.warning(f"Could not compute FLOPs: {e}")
        results_df['flops_g'] = np.nan

    return results_df


def analyze_results(results_dir: str, config: dict):
    """Aggregate all parquet result files in *results_dir* and generate plots & JSON summary."""
    all_files = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if f.endswith('.parquet')]
    if not all_files:
        logging.error("No result files found in directory.")
        return {}

    df = pd.concat([pd.read_parquet(f) for f in all_files])

    # --- Final Performance Aggregation ---
    final_metrics = df.loc[df.groupby(['experiment_id'])['frame'].idxmax()]
    agg_results = final_metrics.groupby(['method', 'model_name', 'dataset_name', 'corruption', 'eta']).agg(
        mean_acc=('online_acc', 'mean'),
        std_acc=('online_acc', 'std'),
        mean_latency=('latency_ms', 'mean'),
        mean_power=('power_w', 'mean')
    ).reset_index()

    print("\n--- AGGREGATED RESULTS ---")
    print(agg_results.to_string())

    # --- Statistical Tests (Exp 1 Success Criteria) ---
    exp1_df = agg_results[agg_results['model_name'].str.contains('resnet50|vit_base', regex=True)]
    zorro_results = exp1_df[exp1_df['method'] == 'ZORROpp']
    baseline_results = exp1_df[exp1_df['method'] != 'ZORROpp']
    best_baseline = baseline_results.loc[baseline_results.groupby(['model_name', 'dataset_name', 'corruption', 'eta'])['mean_acc'].idxmax()]

    comparison = pd.merge(zorro_results, best_baseline, on=['model_name', 'dataset_name', 'corruption', 'eta'], suffixes=('_zorro', '_baseline'))
    comparison['delta_acc'] = comparison['mean_acc_zorro'] - comparison['mean_acc_baseline']

    print("\n--- ZORRO++ vs Best Baseline ---")
    if not comparison.empty:
        print(comparison[['model_name', 'corruption', 'eta', 'mean_acc_zorro', 'mean_acc_baseline', 'delta_acc']].to_string())
    else:
        print("No comparison available (baseline or ZORRO++ missing in aggregated results).")

    # --- Plotting ---
    plt.figure(figsize=(12, 7))
    sns.barplot(data=exp1_df, x='method', y='mean_acc', hue='corruption')
    plt.title('Experiment 1: Final Accuracy Across Methods and Corruptions')
    plt.ylabel('Online Accuracy (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'exp1_final_accuracy.pdf'))
    plt.close()

    # Exp 2: Bound verification plot
    exp2_df = df[df['experiment_id'].str.contains('exp2')].dropna(subset=['bound_mu'])
    if not exp2_df.empty:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=exp2_df, x='frame', y='bound_mu', label='Theoretical Bound B_t(μ)')
        plt.title('Experiment 2: Finite-Sample Bound for Mean Estimation')
        plt.xlabel('Frame')
        plt.ylabel('L_inf Error Bound')
        plt.loglog()
        plt.grid(True, which="both", ls="--")
        plt.savefig(os.path.join(IMAGES_DIR, 'exp2_bound_verification.pdf'))
        plt.close()

    # --- Final JSON Output ---
    final_summary = {
        'aggregated_performance': agg_results.to_dict('records'),
        'zorro_vs_baseline': comparison.to_dict('records')
    }
    return final_summary


def run_experiments(config: dict):
    """Run the list of experiments specified in *config* and perform final analysis."""
    results_dir = f".research/iteration14/results_{time.strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(results_dir, exist_ok=True)

    from .preprocess import get_data_stream

    for i, exp_config in enumerate(config['experiments']):
        exp_id = (
            f"{exp_config['experiment']}_"
            f"{exp_config['model']['name']}_"
            f"{exp_config['dataset']['name']}_"
            f"{exp_config['dataset'].get('corruption', 'none')}_"
            f"{exp_config['method']}_eta{exp_config['stream']['eta']}_"
            f"seed{exp_config['seed']}"
        )
        logging.info(f"\n--- Starting run [{i+1}/{len(config['experiments'])}]: {exp_id} ---")

        try:
            dataloader = get_data_stream(exp_config, config.get('data_manifest', {}))
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            results_df = run_single_experiment(exp_config, dataloader, device)

            # Add metadata columns
            results_df['experiment_id'] = exp_id
            results_df['method'] = exp_config['method']
            results_df['model_name'] = exp_config['model']['name']
            results_df['dataset_name'] = exp_config['dataset']['name']
            results_df['corruption'] = exp_config['dataset'].get('corruption', 'none')
            results_df['eta'] = exp_config['stream']['eta']
            results_df['seed'] = exp_config['seed']

            output_path = os.path.join(results_dir, f"{exp_id}.parquet")
            results_df.to_parquet(output_path)
            logging.info(f"Saved results for {exp_id} to {output_path}")

        except Exception as e:
            logging.error(f"Experiment {exp_id} failed: {e}", exc_info=True)
            # Continue to next experiment

    # --- Final Analysis and Reporting ---
    final_json_results = analyze_results(results_dir, config)

    print("\n" + "=" * 50)
    print("           FINAL EXPERIMENT SUMMARY (JSON)")
    print("=" * 50 + "\n")
    print(json.dumps(final_json_results, indent=2))

    summary_path = os.path.join(JSON_DIR, f"summary_{time.strftime('%Y%m%d-%H%M%S')}.json")
    os.makedirs(JSON_DIR, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(final_json_results, f, indent=2)
    logging.info(f"Final JSON summary saved to {summary_path}")