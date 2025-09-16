import os
import re
import json
from pathlib import Path
import pandas as pd
import tiktoken
from datasets import load_dataset
import requests
from bs4 import BeautifulSoup
import time

# --- Constants ---
TOKENIZER = tiktoken.get_encoding('cl100k_base')
MAX_TOKENS = 2048

# --- Text Processing Functions ---

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # More robust PHI scrubber
    text = re.sub(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', '<DATE>', text) # Dates
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '<SSN>', text) # SSN
    text = re.sub(r'\[\*\*(.*?)\*\*\]', '<DEIDENTIFIED>', text) # MIMIC format
    text = re.sub(r'<NAME>|<DOB>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_and_truncate(text):
    tokens = TOKENIZER.encode(text)
    if len(tokens) > MAX_TOKENS:
        tokens = tokens[:MAX_TOKENS]
    return TOKENIZER.decode(tokens)

# --- Data Acquisition Functions ---

def download_mimic_iv_discharge(config, output_dir):
    # MIMIC-IV requires user credentials and is not directly downloadable.
    # This function checks for its existence and provides instructions if absent.
    mimic_path = Path(config['paths']['mimic_iv_dir']) / 'discharge.csv'
    if not mimic_path.exists():
        raise FileNotFoundError(
            f"FATAL: MIMIC-IV data not found at '{mimic_path}'. "
            "This experiment requires the 'discharge.csv' file from the MIMIC-IV 'note' module. "
            "Please download it from PhysioNet (https://physionet.org/content/mimiciv/2.2/) and place it in the directory specified by 'paths.mimic_iv_dir' in your config. "
            "NO-FALLBACK RULE: Cannot proceed without this real dataset."
        )
    print(f"Found MIMIC-IV data at {mimic_path}. Processing...")
    df = pd.read_csv(mimic_path, usecols=['text'])
    df = df.rename(columns={'text': 'prompt'})
    df['task'] = 'HIPAA-MedTriage'
    return df.dropna()

def scrape_sec_edgar_10k_risk_factors(config, output_dir):
    print("Scraping SEC EDGAR 10-K filings for 'Risk Factors'...")
    headers = {'User-Agent': 'Research Co. researcher@example.com'}
    ciks = config['sec_edgar_ciks']
    all_risk_factors = []

    for cik in ciks:
        try:
            # Get the list of filings
            url = f'https://data.sec.gov/submissions/CIK{cik}.json'
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            filings = response.json()['filings']['recent']
            time.sleep(0.2) # Rate limit
            
            # Find the most recent 10-K
            ten_k_accession = None
            for acc_num, form in zip(filings['accessionNumber'], filings['form']):
                if form == '10-K':
                    ten_k_accession = acc_num.replace('-', '')
                    primary_doc = filings['primaryDocument'][filings['accessionNumber'].index(acc_num)]
                    break
            
            if not ten_k_accession:
                print(f"Warning: No 10-K found for CIK {cik}. Skipping.")
                continue

            # Get the 10-K HTML
            filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{ten_k_accession}/{primary_doc}"
            response = requests.get(filing_url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the 'Risk Factors' section (this is heuristic)
            risk_header = soup.find(lambda tag: 'risk factors' in tag.get_text(strip=True).lower() and tag.name in ['h1','h2','h3','b'])
            if risk_header:
                content = []
                for sibling in risk_header.find_next_siblings():
                    # Stop at the next major header
                    if sibling.name in ['h1', 'h2', 'h3']:
                        break
                    if sibling.name == 'p':
                        content.append(sibling.get_text(strip=True))
                full_text = ' '.join(content)
                if full_text:
                    # Split into smaller prompt-like chunks
                    sentences = re.split(r'(?<=[.!?]) +', full_text)
                    chunks = [' '.join(sentences[i:i+5]) for i in range(0, len(sentences), 5)]
                    all_risk_factors.extend(chunks)
        except Exception as e:
            print(f"Warning: Failed to process CIK {cik}. Error: {e}")
    
    if not all_risk_factors:
         raise RuntimeError("FATAL: Failed to scrape any risk factors from SEC EDGAR. Cannot proceed.")

    df = pd.DataFrame(all_risk_factors, columns=['prompt'])
    df['task'] = 'FinReg-Compliance'
    return df

def download_hf_dataset(name, path, split, col, task_name):
    print(f"Downloading Hugging Face dataset: {name}")
    try:
        dataset = load_dataset(path, split=split)
        df = dataset.to_pandas()
        df = df.rename(columns={col: 'prompt'})
        df['task'] = task_name
        return df[['prompt', 'task']]
    except Exception as e:
        raise ConnectionError(f"FATAL: Could not download Hugging Face dataset '{path}'. Error: {e}")

# --- Main Preprocessing Script ---

def run_preprocessing(config):
    output_path = Path(config['paths']['processed_data_path'])
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Starting preprocessing. Output will be saved to {output_path}")

    exp_id = config['experiment_id']
    all_tasks_df = pd.DataFrame()

    if exp_id == 1:
        df_mimic = download_mimic_iv_discharge(config, output_path)
        df_sec = scrape_sec_edgar_10k_risk_factors(config, output_path)
        all_tasks_df = pd.concat([df_mimic.head(1500), df_sec.head(1500)], ignore_index=True)

    elif exp_id == 2:
        all_tasks_df = download_hf_dataset('Super-NI', 'allenai/super_natural_instructions', 'train[:5%]', 'definition', 'Super-NI')
    
    elif exp_id == 3:
        df_gsm8k = download_hf_dataset('GSM8K', 'openai/gsm8k', 'train', 'question', 'GSM8K')
        df_creative = download_hf_dataset('BIG-bench', 'google/bigbench', 'reasoning_about_colored_objects', 'inputs', 'creative_story')
        all_tasks_df = pd.concat([df_gsm8k, df_creative], ignore_index=True)

    if all_tasks_df.empty:
        raise ValueError("FATAL: No data was processed. Check experiment config and data sources.")

    print("\nCleaning and tokenizing text...")
    all_tasks_df['prompt'] = all_tasks_df['prompt'].apply(clean_text).apply(tokenize_and_truncate)
    all_tasks_df.dropna(subset=['prompt'], inplace=True)
    all_tasks_df = all_tasks_df[all_tasks_df['prompt'].str.len() > 20] # Remove very short prompts

    # Split data for each task
    opt_dfs, test_dfs, meta_dfs = [], [], []
    for task_name, group in all_tasks_df.groupby('task'):
        print(f"  Splitting task: {task_name} ({len(group)} prompts)")
        # Create a deterministic split
        group = group.sample(frac=1, random_state=42).reset_index(drop=True)
        
        test_split = group.head(50)
        remaining = group.iloc[50:]
        opt_split = remaining.head(300)
        meta_split = remaining.iloc[300:800] # For MetaBO

        test_dfs.append(test_split)
        opt_dfs.append(opt_split)
        if not meta_split.empty: meta_dfs.append(meta_split)

    # --- Combine and Save ---
    final_optimization_df = pd.concat(opt_dfs, ignore_index=True)
    final_test_df = pd.concat(test_dfs, ignore_index=True)
    
    opt_path = output_path / 'optimization.parquet'
    test_path = output_path / 'test.parquet'
    final_optimization_df.to_parquet(opt_path)
    final_test_df.to_parquet(test_path)

    print(f"\nPreprocessing complete.")
    print(f"  Optimization data: {len(final_optimization_df)} samples saved to {opt_path}")
    print(f"  Test data: {len(final_test_df)} samples saved to {test_path}")

    if meta_dfs:
        final_meta_df = pd.concat(meta_dfs, ignore_index=True)
        meta_path = output_path / 'meta_train.parquet'
        final_meta_df.to_parquet(meta_path)
        print(f"  Meta-training data: {len(final_meta_df)} samples saved to {meta_path}")
    
    return str(output_path)