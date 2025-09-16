import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')

def load_results(results_directory):
    all_dfs = []
    for filename in os.listdir(results_directory):
        if filename.endswith('.csv'):
            try:
                df = pd.read_csv(os.path.join(results_directory, filename))
                parts = filename.replace('.csv', '').split('_')
                df['exp_name'] = parts[0]
                df['model'] = parts[1]
                df['dataset'] = parts[2]
                df['corruption'] = parts[3]
                df['method'] = parts[4]
                df['eta'] = float(parts[5].replace('eta', ''))
                df['seed'] = int(parts[6].replace('seed', ''))
                all_dfs.append(df)
            except Exception as e:
                print(f"Could not parse {filename}: {e}")
    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)

def analyze_experiment_1(df, output_dir):
    print("\n--- Analyzing Experiment 1: 4th-order Moment Normalization ---")
    if df.empty:
        print("No data for Experiment 1.")
        return {}

    # Final accuracy for each run
    final_acc = df.loc[df.groupby(['exp_name', 'model', 'dataset', 'corruption', 'method', 'eta', 'seed'])['frame'].idxmax()]
    
    # Average over seeds
    summary = final_acc.groupby(['model', 'dataset', 'corruption', 'method', 'eta'])['accuracy'].agg(['mean', 'std']).reset_index()

    # Plot: Accuracy vs. Method for a key corruption
    for dataset in summary['dataset'].unique():
        for eta in summary['eta'].unique():
            plt.figure(figsize=(12, 7))
            subset = summary[(summary['dataset'] == dataset) & (summary['eta'] == eta) & (summary['corruption'] == 'shot_noise')]
            if subset.empty:
                continue
            sns.barplot(data=subset, x='method', y='mean', hue='model')
            plt.title(f'Accuracy on {dataset} (shot_noise, eta={eta})')
            plt.ylabel('Online Top-1 Accuracy (%)')
            plt.xlabel('Method')
            plt.xticks(rotation=45)
            plt.tight_layout()
            filename = os.path.join(output_dir, f'exp1_accuracy_{dataset}_eta{eta}.pdf')
            plt.savefig(filename)
            plt.close()
            print(f"Saved plot: {filename}")

    # Calculate DeltaAcc_HT-C
    htc_summary = summary[summary['corruption'].str.contains('htc')]
    if not htc_summary.empty:
        zorropp_acc = htc_summary[htc_summary['method'] == 'ZorroPP'].set_index(['model', 'dataset', 'eta'])['mean']
        baselines_max_acc = htc_summary[htc_summary['method'] != 'ZorroPP'].groupby(['model', 'dataset', 'eta'])['mean'].max()
        delta_acc = (zorropp_acc - baselines_max_acc).reset_index()
        print("\nDelta Accuracy on HT-C corruptions:")
        print(delta_acc)
        return delta_acc.to_dict('records')
    return {}

def analyze_experiment_2(df, output_dir):
    print("\n--- Analyzing Experiment 2: Finite-Sample Bounds ---")
    # This requires custom log files from train.py not implemented in this version.
    # Generating placeholder analysis.
    print("Analysis for Experiment 2 requires ground truth moment logs, which are not generated in this version.")
    return {'status': 'Skipped, requires custom logging'}

def analyze_experiment_3(df, output_dir):
    print("\n--- Analyzing Experiment 3: Edge Device Scheduling & Privacy ---")
    if df.empty or df[df['exp_name'] != 'exp3'].empty:
        print("No data for Experiment 3.")
        return {}
    
    exp3_df = df[df['exp_name'] == 'exp3'].copy()
    exp3_df['variant'] = exp3_df['method'] # Using method as variant name

    summary = exp3_df.groupby('variant').agg(
        mean_accuracy=pd.NamedAgg(column='accuracy', aggfunc='last'),
        mean_latency_ms=pd.NamedAgg(column='latency_ms', aggfunc='mean'),
        p99_latency_ms=pd.NamedAgg(column='latency_ms', aggfunc=lambda x: x.quantile(0.995)),
        mean_power_watts=pd.NamedAgg(column='power_watts', aggfunc='mean'),
        p99_power_watts=pd.NamedAgg(column='power_watts', aggfunc=lambda x: x.quantile(0.995)),
    ).reset_index()

    print("\nEdge Performance Summary:")
    print(summary)

    # Plot time series data
    for metric in ['accuracy', 'latency_ms', 'power_watts']:
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=exp3_df, x='frame', y=metric, hue='variant')
        plt.title(f'{metric.replace("_", " ").title()} over Time on EdgeHAR-C')
        plt.xlabel('Frame')
        plt.ylabel(metric)
        filename = os.path.join(output_dir, f'exp3_timeseries_{metric}.pdf')
        plt.savefig(filename)
        plt.close()
        print(f"Saved plot: {filename}")
        
    return summary.to_dict('records')

def generate_report(results_directory):
    print(f"Generating report from results in: {results_directory}")
    output_dir = os.path.join(results_directory, 'analysis_plots')
    os.makedirs(output_dir, exist_ok=True)

    full_df = load_results(results_directory)
    if full_df.empty:
        print("No result files found. Exiting analysis.")
        final_results = {'error': 'No result files found.'}
    else:
        final_results = {
            'experiment_1': analyze_experiment_1(full_df[full_df['exp_name'] == 'exp1'], output_dir),
            'experiment_2': analyze_experiment_2(full_df[full_df['exp_name'] == 'exp2'], output_dir),
            'experiment_3': analyze_experiment_3(full_df[full_df['exp_name'] == 'exp3'], output_dir),
        }

    # Output final summary to JSON file and stdout
    summary_path = os.path.join(results_directory, 'final_results.json')
    with open(summary_path, 'w') as f:
        json.dump(final_results, f, indent=4)
    
    print("\n========================================")
    print("======= FINAL RESULTS SUMMARY ========")
    print("========================================")
    print(json.dumps(final_results, indent=2))

    return final_results
