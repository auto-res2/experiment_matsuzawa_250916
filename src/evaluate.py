import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel

IMAGES_DIR = os.path.join(".research", "iteration8", "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

RESULTS_DIR = os.path.join(".research", "iteration8")
os.makedirs(RESULTS_DIR, exist_ok=True)

def analyze_experiment(exp_name, exp_config, base_dir):
    results = {}
    all_seeds_data = []

    for model_name in exp_config['models']:
        model_results = []
        for seed in exp_config['seeds']:
            log_path = os.path.join(base_dir, f"{exp_name}_{model_name}_seed{seed}", "train_log.json")
            if not os.path.exists(log_path):
                print(f"Warning: Log file not found for {exp_name}/{model_name}/seed{seed}")
                continue
            with open(log_path, "r") as f:
                data = json.load(f)
                model_results.append(data)
                all_seeds_data.append({"model": model_name, "seed": seed, **data})

        if not model_results:
            continue

        # Aggregate results across seeds
        final_test_acc = [max([e['test_acc'] for e in r['log']]) for r in model_results]
        time_to_convergence = []
        for r in model_results:
            target_acc = 0.75 * max([e['val_acc'] for e in r['log']])
            converged_epoch = next((e for e in r['log'] if e['val_acc'] >= target_acc), None)
            if converged_epoch:
                time_taken = sum([e['epoch_time_s'] for e in r['log'] if e['epoch'] <= converged_epoch['epoch']])
                time_to_convergence.append(time_taken / 3600)  # to GPU-hours
            else:
                time_to_convergence.append(float("inf"))

        avg_epoch_time = [np.mean([e['epoch_time_s'] for e in r['log']]) for r in model_results]
        peak_gpu_mem = [max([e['gpu_mem_gb'] for e in r['log']]) for r in model_results]
        total_energy = [r.get('total_energy_kWh', 0) for r in model_results]
        warmup_epochs = []
        for r in model_results:
            warmup_epoch = next((e['epoch'] for e in r['log'] if e['predictor_corr_rho'] > 0.5), float("inf"))
            warmup_epochs.append(warmup_epoch)

        results[model_name] = {
            'final_test_acc_mean': np.mean(final_test_acc),
            'final_test_acc_std': np.std(final_test_acc),
            'gpu_hours_to_75_acc_mean': np.mean(time_to_convergence),
            'gpu_hours_to_75_acc_std': np.std(time_to_convergence),
            'avg_epoch_time_s_mean': np.mean(avg_epoch_time),
            'avg_epoch_time_s_std': np.std(avg_epoch_time),
            'peak_gpu_mem_gb_mean': np.mean(peak_gpu_mem),
            'peak_gpu_mem_gb_std': np.std(peak_gpu_mem),
            'total_energy_kWh_mean': np.mean(total_energy),
            'total_energy_kWh_std': np.std(total_energy),
            'warmup_epochs_rho_gt_0.5_mean': np.mean(warmup_epochs),
        }
    return results, all_seeds_data

def run_statistical_tests(results_df):
    models = results_df['model'].unique()
    if len(models) < 2:
        return

    print("\n--- Paired t-tests (p < 0.05) ---")
    if 'meta-leap' in models:
        meta_leap_data = results_df[results_df['model'] == 'meta-leap'].sort_values('seed')
        for model_name in models:
            if model_name == 'meta-leap':
                continue
            compare_data = results_df[results_df['model'] == model_name].sort_values('seed')
            if len(meta_leap_data) != len(compare_data):
                print(f"Cannot compare meta-leap and {model_name}: mismatched seed counts")
                continue
            merged = pd.merge(meta_leap_data, compare_data, on='seed', suffixes=('_meta', '_other'))
            for metric in ['final_test_acc', 'gpu_hours_to_75_acc', 'avg_epoch_time_s']:
                if f"{metric}_meta" in merged and f"{metric}_other" in merged:
                    stat, p_val = ttest_rel(merged[f"{metric}_meta"], merged[f"{metric}_other"])
                    if p_val < 0.05:
                        print(
                            f"Metric '{metric}': meta-leap is significantly different from {model_name} (p={p_val:.4f})"
                        )


def create_plots(exp_name, all_seeds_data):
    df = pd.DataFrame(all_seeds_data)
    if df.empty:
        return

    # Unroll the logs
    records = []
    for _, row in df.iterrows():
        for log_entry in row['log']:
            records.append({"model": row['model'], "seed": row['seed'], **log_entry})
    log_df = pd.DataFrame(records)
    if log_df.empty:
        return

    plt.style.use("seaborn-v0_8-whitegrid")

    # Plot 1: Validation Accuracy vs. Wall-clock time
    log_df['cumulative_time_s'] = log_df.groupby(['model', 'seed'])['epoch_time_s'].cumsum()
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=log_df,
        x='cumulative_time_s',
        y='val_acc',
        hue='model',
        errorbar='sd',
    )
    plt.title(f"{exp_name}: Validation Accuracy vs. Time")
    plt.xlabel("Wall-clock Time (seconds)")
    plt.ylabel("Validation Accuracy")
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, f"{exp_name}_accuracy_vs_time.pdf"))
    plt.close()

    # Plot 2: Predictor Correlation (Rho) vs. Epoch
    if 'predictor_corr_rho' in log_df.columns:
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=log_df,
            x='epoch',
            y='predictor_corr_rho',
            hue='model',
            errorbar='sd',
        )
        plt.title(f"{exp_name}: Predictor Correlation ($\\rho$) vs. Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Pearson Correlation ($\\rho$)")
        plt.axhline(y=0.5, color='r', linestyle='--', label="$\\rho=0.5$")
        plt.legend(title="Model")
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGES_DIR, f"{exp_name}_rho_vs_epoch.pdf"))
        plt.close()


def main(config):
    base_dir = os.path.join(".research", "iteration1")
    all_results = {}

    for exp_name, exp_config in config['experiments'].items():
        print(f"\n--- Evaluating Experiment: {exp_name} ---")

        results, all_seeds_data = analyze_experiment(exp_name, exp_config, base_dir)
        all_results[exp_name] = results

        # Save per-experiment JSON to mandated directory
        json_path = os.path.join(RESULTS_DIR, f"{exp_name}_summary.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Saved summary to {json_path}")

        # Create plots
        create_plots(exp_name, all_seeds_data)

        # Statistical tests
        if all_seeds_data:
            df_list = []
            for item in all_seeds_data:
                best_epoch = max(item['log'], key=lambda x: x['val_acc'])
                time_to_75 = float("inf")
                target_acc = 0.75 * best_epoch['val_acc']
                converged = next((e for e in item['log'] if e['val_acc'] >= target_acc), None)
                if converged:
                    time_to_75 = (
                        sum(e['epoch_time_s'] for e in item['log'] if e['epoch'] <= converged['epoch'])
                        / 3600
                    )

                df_list.append(
                    {
                        'model': item['model'],
                        'seed': item['seed'],
                        'final_test_acc': best_epoch['test_acc'],
                        'gpu_hours_to_75_acc': time_to_75,
                        'avg_epoch_time_s': np.mean([e['epoch_time_s'] for e in item['log']]),
                    }
                )
            run_statistical_tests(pd.DataFrame(df_list))

    # Final JSON output (aggregated)
    final_path = os.path.join(RESULTS_DIR, "final_summary.json")
    with open(final_path, "w") as f:
        json.dump(all_results, f, indent=4)

    print("\n--- Final Aggregated Results ---")
    print(json.dumps(all_results, indent=4))

    # Machine-parseable output for automated checking
    print("\n--- Machine-Parseable Output ---")
    print(json.dumps(all_results))