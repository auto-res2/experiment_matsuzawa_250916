import json
import time
import argparse
from pathlib import Path
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch
from transformers import AutoModel, AutoTokenizer

# --- Metric Calculation Functions -------------------------------------------------
# (unchanged content up to path modifications)
# ...

# --- Plotting Functions -----------------------------------------------------------
# (unchanged)

# --- GUI Function -----------------------------------------------------------
# (unchanged)

# --- Main Evaluation Script -------------------------------------------------------

def run_evaluation(config, training_run_dir):
    training_run_path = Path(training_run_dir)
    result_files = sorted(training_run_path.glob("*.json"))
    if not result_files:
        raise FileNotFoundError(
            f"No training result files (*.json) found in '{training_run_path}'"
        )
    latest_result_file = result_files[-1]

    print(f"\nEvaluating results from: {latest_result_file}")
    with open(latest_result_file, 'r') as f:
        results = json.load(f)

    exp_id = config["experiment_id"]
    final_report = {'experiment_id': exp_id, 'config_name': config['name']}
    figure_paths = []

    # Mandatory paths for iteration15
    output_dir_img = Path(".research/iteration15/images")
    output_dir_img.mkdir(parents=True, exist_ok=True)
    output_dir_json = Path(".research/iteration15/")
    output_dir_json.mkdir(parents=True, exist_ok=True)

    if exp_id == 1:
        metrics = calculate_exp1_metrics(results, config)
        final_report['results'] = metrics
        reflect_h60 = [max(s['helpfulness'] for s in h) for h in results.get('REFLECT-BO', [])]
        shift_h60 = [max(s['helpfulness'] for s in h) for h in results.get('SHIFT-BO', [])]
        if reflect_h60 and shift_h60:
            stat, p = stats.wilcoxon(reflect_h60, shift_h60)
            final_report['statistical_tests'] = {
                'wilcoxon_reflect_vs_shift_h60': {'statistic': stat, 'p_value': p}
            }
        fig_path = plot_performance_curves(results, output_dir_img, exp_id)
        figure_paths.append(fig_path)

    elif exp_id == 2:
        drift_metrics = calculate_exp2_metrics(results)
        mia_metrics = run_membership_inference_attack(config, training_run_dir)
        final_report['results'] = {**drift_metrics, **mia_metrics}

    elif exp_id == 3:
        final_report['results'] = calculate_exp3_metrics(results)
        print("\nTo view the simulated GUI for Experiment 3, run:")
        print("streamlit run src/evaluate.py -- --gui")

    carbon_files = sorted(training_run_path.glob("*.csv"))
    if carbon_files:
        df_carbon = pd.read_csv(carbon_files[-1])
        total_emissions_kg = df_carbon['emissions (kg)'].sum()
        final_report['carbon_footprint_kg_co2eq'] = total_emissions_kg

    ts = int(time.time())
    json_output_path = output_dir_json / f"evaluation_report_exp{exp_id}_{ts}.json"
    with open(json_output_path, "w") as f:
        json.dump(final_report, f, indent=4)

    print("\n--- Evaluation Report Summary ---")
    print(json.dumps(final_report, indent=4))
    print(f"\nFull report saved to: {json_output_path}")
    print("Generated figures:", figure_paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="Launch the Streamlit GUI for Exp 3.")
    args = parser.parse_args()

    if args.gui:
        run_gui(None)