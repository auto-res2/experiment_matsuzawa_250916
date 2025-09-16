import os
import re
from pathlib import Path
import pandas as pd
import tiktoken
from datasets import load_dataset, concatenate_datasets

# --- Constants ---
TOKENIZER = tiktoken.get_encoding('cl100k_base')
MAX_TOKENS = 2048

# --- Helper Functions ---
def clean_text(text):
    """Lowercases and scrubs PHI markers."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Rule-based PHI scrubber
    phi_patterns = [
        r'<name>', r'</name>',
        r'<dob>', r'</dob>',
        r'<ssn>', r'</ssn>',
        r'<phone>', r'</phone>',
        r'<address>', r'</address>'
    ]
    for pattern in phi_patterns:
        text = re.sub(pattern, '', text)
    return text.strip()

def tokenize_and_truncate(text):
    """Tokenizes text and truncates to MAX_TOKENS."""
    tokens = TOKENIZER.encode(text)
    if len(tokens) > MAX_TOKENS:
        tokens = tokens[:MAX_TOKENS]
    # This part is illustrative; we save the text, not tokens, for BO.
    return TOKENIZER.decode(tokens)

def load_secure_data(path, dataset_name):
    """Loads secure data, raising an error if it doesn't exist."""
    file_path = Path(path) / f"{dataset_name}.json"
    if not file_path.exists():
        raise FileNotFoundError(
            f"FATAL: Secure dataset '{dataset_name}.json' not found at '{file_path}'. "
            f"This experiment requires real, private data. NO FALLBACK to dummy data is allowed. "
            f"Please ensure the file exists and the path is correctly configured."
        )
    print(f"Securely loading data from {file_path}...")
    # Assuming JSONL format with 'prompt' and 'is_test_prompt' keys
    df = pd.read_json(file_path, lines=True)
    return df

def process_dataframe(df, prompt_col, is_test_col=None):
    df['prompt'] = df[prompt_col].apply(clean_text).apply(tokenize_and_truncate)
    df = df[['prompt']]
    if is_test_col:
        optimization_df = df[~df[is_test_col]].reset_index(drop=True)
        test_df = df[df[is_test_col]].reset_index(drop=True)
    else: # If no split info, create a random 80/20 split
        test_df = df.sample(frac=0.2, random_state=42)
        optimization_df = df.drop(test_df.index)
    return optimization_df, test_df

# --- Main Preprocessing Script ---
def run(config):
    """Main preprocessing entry point."""
    output_path = Path(config['paths']['processed_data_path'])
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Starting preprocessing. Output will be saved to {output_path}")

    all_optimization_dfs = []
    all_test_dfs = []

    # --- Experiment 1: Secure, Data-Scarce Datasets ---
    if config['experiment_to_run'] == 1:
        print("\n--- Processing datasets for Experiment 1 ---")
        secure_path = config['paths'].get('secure_data_path')
        if not secure_path:
             raise ValueError("FATAL: 'secure_data_path' is not defined in the configuration for Experiment 1.")

        for name in config['experiment_1']['datasets']:
            df = load_secure_data(secure_path, name)
            # Assume a column 'is_test_prompt' (boolean) exists for splitting
            if 'is_test_prompt' not in df.columns:
                 raise ValueError(f"Secure dataset {name} must contain 'is_test_prompt' column.")
            opt_df, test_df = process_dataframe(df, 'prompt', 'is_test_prompt')
            all_optimization_dfs.append(opt_df)
            all_test_dfs.append(test_df)

    # --- Experiment 2: Public Datasets Stream ---
    elif config['experiment_to_run'] == 2:
        print("\n--- Processing datasets for Experiment 2 ---")
        # Using Super-Natural-Instructions
        dataset = load_dataset("allenai/super_natural_instructions", split='train')
        df = dataset.to_pandas().rename(columns={'instruction': 'prompt'})
        # Take a subset for manageable processing
        df = df.sample(n=config['experiment_2']['total_prompts'], random_state=42)
        opt_df, test_df = process_dataframe(df, 'prompt')
        all_optimization_dfs.append(opt_df)
        all_test_dfs.append(test_df)

    # --- Experiment 3: Mixed Public Datasets ---
    elif config['experiment_to_run'] == 3:
        print("\n--- Processing datasets for Experiment 3 ---")
        datasets_to_load = {
            'gsm8k': ('openai/gsm8k', 'main', 'question'),
            'aqua_rat': ('deepmind/aqua_rat', 'raw', 'question'),
            'cnn_dailymail': ('abisee/cnn_dailymail', '3.0.0', 'article'),
        }
        for name, (path, subset, col) in datasets_to_load.items():
            print(f"Loading {name}...")
            dataset = load_dataset(path, subset, split='test') # Use test split for smaller size
            df = dataset.to_pandas().rename(columns={col: 'prompt'})
            opt_df, test_df = process_dataframe(df, 'prompt')
            all_optimization_dfs.append(opt_df)
            all_test_dfs.append(test_df)

    # --- Combine and Save ---
    if not all_optimization_dfs or not all_test_dfs:
        raise ValueError("No data was processed. Check experiment configuration and data availability.")

    final_optimization_df = pd.concat(all_optimization_dfs, ignore_index=True)
    final_test_df = pd.concat(all_test_dfs, ignore_index=True)

    opt_path = output_path / 'optimization.parquet'
    test_path = output_path / 'test.parquet'

    final_optimization_df.to_parquet(opt_path)
    final_test_df.to_parquet(test_path)

    print(f"\nPreprocessing complete.")
    print(f"Optimization data: {len(final_optimization_df)} samples saved to {opt_path}")
    print(f"Test data: {len(final_test_df)} samples saved to {test_path}")
    
    return str(output_path)