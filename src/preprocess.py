import os
import re
import time
import unicodedata
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import requests
from bs4 import BeautifulSoup
from datasets import load_dataset, DownloadConfig
import tiktoken
import numpy as np  # Added import for numpy

# --- Constants ---
MAX_TOKENS = 2048
TOKENIZER = tiktoken.get_encoding("cl100k_base")

# ----------------------------
# Helper functions
# ----------------------------

def _safe_concat(dfs, ignore_index=True):
    """Concatenate a list of DataFrames, ignoring those that are empty.

    Raises RuntimeError if **all** provided frames are empty.
    """
    non_empty = [df for df in dfs if not df.empty]
    if not non_empty:
        raise RuntimeError("All input DataFrames are empty – cannot concatenate. NO-FALLBACK RULE invoked.")
    return pd.concat(non_empty, ignore_index=ignore_index)

# --- Text Cleaning and Processing ---

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # PHI Scrubbing (simple version)
    text = re.sub(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', '<DATE>', text)
    text = re.sub(r'\[\*\*(.*?)\*\*\]', '<DEIDENTIFIED>', text)
    # Unicode normalization, lowercasing, and space collapse
    text = unicodedata.normalize('NFKC', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize_and_truncate(text: str) -> dict:
    tokens = TOKENIZER.encode(text)
    truncated = len(tokens) > MAX_TOKENS
    if truncated:
        tokens = tokens[:MAX_TOKENS]
    return {'text': TOKENIZER.decode(tokens), 'truncated': truncated}

# --- Data Acquisition ---

def acquire_finreg_compliance_data(config: dict) -> pd.DataFrame:
    """Scrapes SEC EDGAR for 10-K 'Risk Factors' sections."""
    ciks = config.get('sec_edgar_ciks', [])
    if not ciks:
        return pd.DataFrame()

    print("Acquiring FinReg-Compliance data from SEC EDGAR...")
    headers = {'User-Agent': 'research@example.com'}
    all_prompts = []
    for cik in ciks:
        try:
            time.sleep(0.2)  # Rate limit
            # Find the latest 10-K filing
            url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
            res = requests.get(url, headers=headers)
            res.raise_for_status()
            filings = res.json()['filings']['recent']
            ten_k_indices = [i for i, form in enumerate(filings['form']) if form == '10-K']
            if not ten_k_indices:
                continue

            latest_filing_idx = ten_k_indices[0]
            acc_no = filings['accessionNumber'][latest_filing_idx]
            doc_name = filings['primaryDocument'][latest_filing_idx]

            # Fetch the filing document
            filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_no.replace('-', '')}/{doc_name}"
            doc_res = requests.get(filing_url, headers=headers)
            doc_res.raise_for_status()
            soup = BeautifulSoup(doc_res.content, 'html.parser')

            # Extract text from risk factors section (very basic extraction)
            text = soup.get_text(separator=' ')
            match = re.search(r"(?i)risk factors(.+?)item 1b", text, re.DOTALL)
            if match:
                # Split into smaller prompt-sized chunks
                risk_text = match.group(1).strip()
                sentences = re.split(r'(?<=[.!?]) +', risk_text)
                prompts = [' '.join(sentences[i : i + 5]) for i in range(0, len(sentences), 5)]
                all_prompts.extend(prompts)
        except Exception as e:
            print(f"Warning: Could not scrape data for CIK {cik}. Reason: {e}")

    if not all_prompts:
        raise RuntimeError("Failed to acquire any data from SEC EDGAR. NO-FALLBACK RULE invoked.")

    df = pd.DataFrame({'prompt': all_prompts})
    df['task'] = 'FinReg-Compliance-5'
    return df


def acquire_hipaa_medtriage_data(config: dict) -> pd.DataFrame:
    """Loads local MIMIC-IV discharge summaries, enforcing NO-FALLBACK."""
    print("Attempting to load HIPAA-MedTriage data...")
    mimic_path = Path(config['paths'].get('mimic_iv_dir', ''))
    if not mimic_path.is_dir():
        raise FileNotFoundError(
            f"MIMIC-IV directory not found at '{mimic_path}'. NO-FALLBACK: This dataset is required for the specified experiment."
        )

    # Assuming data is pre-processed into a Parquet file for efficiency
    mimic_file = mimic_path / 'discharge_summaries.parquet'
    if not mimic_file.exists():
        raise FileNotFoundError(
            f"Processed MIMIC-IV file 'discharge_summaries.parquet' not found in '{mimic_path}'. NO-FALLBACK."
        )

    df = pd.read_parquet(mimic_file)
    df = df[['text']].rename(columns={'text': 'prompt'})
    df['task'] = 'HIPAA-MedTriage-5'
    return df


def acquire_hf_dataset(dataset_name: str, config_name: str, split: str, column: str, task: str) -> pd.DataFrame:
    """Loads a dataset from Hugging Face Hub."""
    print(f"Acquiring {dataset_name} from Hugging Face Hub...")
    try:
        hf_token = os.getenv("HF_TOKEN")
        dataset = load_dataset(
            dataset_name,
            config_name,
            split=split,
            token=hf_token,
            download_config=DownloadConfig(use_etag=False),
        )
        df = dataset.to_pandas()
        df = df[[column]].rename(columns={column: 'prompt'})
        df['task'] = task
        return df
    except Exception as e:
        raise ConnectionError(
            f"Failed to download '{dataset_name}' from Hugging Face Hub. NO-FALLBACK. Error: {e}"
        )

# --- Main Execution Logic ---

def run(config: dict):
    output_dir = Path(config['paths']['processed_data_path'])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Preprocessing data for experiment {config['experiment_id']}. Output to: {output_dir}")

    exp_id = config['experiment_id']
    exp_conf = config[f'experiment_{exp_id}']
    tasks = exp_conf['tasks']

    df_list = []
    for task in tasks:
        if task == 'HIPAA-MedTriage-5':
            df_list.append(acquire_hipaa_medtriage_data(config))
        elif task == 'FinReg-Compliance-5':
            df_list.append(acquire_finreg_compliance_data(config))
        elif task == 'Super-NI':
            df_list.append(
                acquire_hf_dataset(
                    'super_natural_instructions',
                    None,
                    'train[:2000]',
                    'definition',
                    'Super-NI',
                )
            )
        elif task == 'GSM8K':
            df_list.append(
                acquire_hf_dataset(
                    'openai/gsm8k',
                    'main',
                    'train',
                    'question',
                    'GSM8K',
                )
            )
        else:
            print(f"Warning: Unknown task '{task}' specified in config. Skipping.")

    if not df_list:
        raise ValueError("No data was loaded. Check task configuration.")

    # Concatenate all tasks safely
    df_all = _safe_concat(df_list)
    print(f"Loaded a total of {len(df_all)} raw prompts.")

    # Clean and tokenize
    print("Cleaning and tokenizing text...")
    df_all['prompt'] = df_all['prompt'].apply(clean_text)
    processed_data = df_all['prompt'].apply(tokenize_and_truncate)
    df_all['prompt'] = [d['text'] for d in processed_data]
    df_all['truncated'] = [d['truncated'] for d in processed_data]
    df_all.dropna(subset=['prompt'], inplace=True)
    df_all = df_all[df_all['prompt'].str.len() > 10]

    if df_all.empty:
        raise RuntimeError("All prompts became empty after cleaning/tokenization. NO-FALLBACK.")

    # Deterministic split
    print("Creating deterministic splits...")
    np.random.seed(42)
    df_all = df_all.sample(frac=1).reset_index(drop=True)

    opt_dfs, test_dfs = [], []
    for task_name, group_df in df_all.groupby('task'):
        test_size = min(50, len(group_df) // 6)  # approx 1/6th for test, capped at 50
        opt_size = min(250, len(group_df) - test_size)

        test_dfs.append(group_df.head(test_size))
        opt_dfs.append(group_df.iloc[test_size : test_size + opt_size])

    # Safe concatenation to avoid "No objects to concatenate" when a split is empty
    df_opt = _safe_concat(opt_dfs)
    df_test = _safe_concat(test_dfs) if any(not df.empty for df in test_dfs) else pd.DataFrame(columns=df_all.columns)

    # Save to Parquet
    opt_path = output_dir / 'optimization.parquet'
    test_path = output_dir / 'test.parquet'
    df_opt.to_parquet(opt_path)
    df_test.to_parquet(test_path)

    print("--- Preprocessing Complete ---")
    print(f"Optimization set: {len(df_opt)} samples saved to {opt_path}")
    print(f"Hold-out test set: {len(df_test)} samples saved to {test_path}")
    return str(output_dir)
