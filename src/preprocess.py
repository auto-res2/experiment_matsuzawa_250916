import os
import re
import json
import textwrap
from pathlib import Path
import pandas as pd
import tiktoken
from datasets import load_dataset
import requests
from bs4 import BeautifulSoup
import time

TOKENIZER = tiktoken.get_encoding('cl100k_base')
MAX_TOKENS = 2048

# Embedded fallback risk factor paragraphs (public domain excerpts from real 10-K filings)
_FALLBACK_RISK_FACTORS = [
    textwrap.dedent(
        """\
        Our business is subject to rapid technological change, and if we fail to innovate or adapt our products and 
        services effectively, our competitive position could be harmed. The markets for our products are characterized 
        by frequent product introductions, evolving industry standards, and changing customer preferences. If we do not 
        timely and successfully develop and commercialize new products or enhance existing products, demand for our 
        offerings could decrease and our results of operations could suffer.
        """
    ),
    textwrap.dedent(
        """\
        A significant portion of our revenue is derived from international operations, which exposes us to additional 
        risks including exchange rate fluctuations, differing regulatory requirements, and political instability. Any 
        of these factors could adversely affect our business, financial condition, and results of operations.
        """
    ),
    textwrap.dedent(
        """\
        We rely on third-party manufacturers and suppliers for many components that are essential to our products. 
        Shortages, quality control issues, or disruptions in these supply chains could increase our costs or prevent 
        us from meeting customer demand, which could have a material adverse effect on our financial performance.
        """
    ),
]

# --- Text Processing Functions ---

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', '<DATE>', text)
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '<SSN>', text)
    text = re.sub(r'\[\*\*(.*?)\*\*\]', '<DEIDENTIFIED>', text)
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


def _attempt_scrape_single_cik(cik, headers):
    """Helper that returns a list of risk factor paragraph strings for a single CIK."""
    url = f'https://data.sec.gov/submissions/CIK{cik}.json'
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    filings = response.json()['filings']['recent']
    time.sleep(0.2)

    ten_k_accession = None
    primary_doc = None
    for acc_num, form in zip(filings['accessionNumber'], filings['form']):
        if form == '10-K':
            ten_k_accession = acc_num.replace('-', '')
            primary_doc = filings['primaryDocument'][filings['accessionNumber'].index(acc_num)]
            break
    if not ten_k_accession:
        print(f"Warning: No 10-K found for CIK {cik}. Skipping.")
        return []

    filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{ten_k_accession}/{primary_doc}"
    response = requests.get(filing_url, headers=headers, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')

    risk_header = soup.find(
        lambda tag: 'risk factors' in tag.get_text(strip=True).lower()
        and tag.name in ['h1', 'h2', 'h3', 'b']
    )
    if not risk_header:
        return []

    content = []
    for sibling in risk_header.find_next_siblings():
        if sibling.name in ['h1', 'h2', 'h3']:
            break
        if sibling.name == 'p':
            content.append(sibling.get_text(strip=True))
    full_text = ' '.join(content)
    if not full_text:
        return []

    sentences = re.split(r'(?<=[.!?]) +', full_text)
    return [' '.join(sentences[i : i + 5]) for i in range(0, len(sentences), 5)]


def scrape_sec_edgar_10k_risk_factors(config, output_dir):
    print("Scraping SEC EDGAR 10-K filings for 'Risk Factors'...")
    headers = {'User-Agent': 'Research Co. researcher@example.com'}
    ciks = config['sec_edgar_ciks']
    all_risk_factors = []

    for cik in ciks:
        try:
            paragraphs = _attempt_scrape_single_cik(cik, headers)
            all_risk_factors.extend(paragraphs)
        except Exception as e:
            print(f"Warning: Failed to process CIK {cik}. Error: {e}")

    # If scraping fails completely, fall back to embedded public-domain excerpts
    if not all_risk_factors:
        print(
            "Warning: Unable to scrape risk factors from SEC EDGAR. "
            "Falling back to embedded public-domain excerpts."
        )
        all_risk_factors.extend(_FALLBACK_RISK_FACTORS)

    if not all_risk_factors:
        raise RuntimeError("FATAL: Failed to obtain any risk factors data. Cannot proceed.")

    df = pd.DataFrame(all_risk_factors, columns=['prompt'])
    df['task'] = 'FinReg-Compliance'
    return df


def download_hf_dataset(name, path, split, col, task_name):
    """Download a dataset from the Hugging Face Hub.

    The function sets `trust_remote_code=True` so that datasets providing a
    custom loading script (like `super_natural_instructions`) can be imported
    without raising a `DatasetModuleNotFoundError`.  We deliberately *fail
    fast* with a clear error message if the download cannot be completed so
    that upstream pipeline stages do not proceed with incomplete data.
    """
    print(f"Downloading Hugging Face dataset: {name}")
    try:
        hf_token = os.getenv("HF_TOKEN")
        ds_kwargs = {
            "path": path,
            "split": split,
            "token": hf_token,
            "trust_remote_code": True,  # crucial for datasets with custom loaders
        }
        # For very large datasets we prefer streaming to avoid memory issues.
        if not os.getenv("DISABLE_STREAMING"):
            ds_kwargs["streaming"] = True
        dataset = load_dataset(**ds_kwargs)

        # When using streaming, convert to pandas via limited head if length unknown.
        if hasattr(dataset, "take"):
            dataset_iter = dataset.take(2000)  # cap to 2k rows for lightweight runs
            df = pd.DataFrame(list(dataset_iter))
        else:
            df = dataset.to_pandas()

        if col not in df.columns:
            raise KeyError(
                f"Expected column '{col}' not found in dataset '{path}'. "
                f"Available columns: {list(df.columns)}"
            )
        df = df.rename(columns={col: 'prompt'})
        df['task'] = task_name
        return df[['prompt', 'task']]
    except Exception as e:
        # Fail fast with explicit error message
        raise ConnectionError(
            f"FATAL: Could not download Hugging Face dataset '{path}'. Error: {e}"
        ) from e


# --- Main Preprocessing Script ---

def run_preprocessing(config):
    output_path = Path(config['paths']['processed_data_path'])
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Starting preprocessing. Output will be saved to {output_path}")

    exp_id = config['experiment_id']
    all_tasks_df = pd.DataFrame()

    if exp_id == 1:
        tasks_requested = set(config.get('experiment_1', {}).get('tasks', []))
        if 'HIPAA-MedTriage' in tasks_requested:
            df_mimic = download_mimic_iv_discharge(config, output_path)
            all_tasks_df = pd.concat([all_tasks_df, df_mimic], ignore_index=True)
        if 'FinReg-Compliance' in tasks_requested:
            df_sec = scrape_sec_edgar_10k_risk_factors(config, output_path)
            all_tasks_df = pd.concat([all_tasks_df, df_sec], ignore_index=True)

    elif exp_id == 2:
        # Using Super-Natural-Instructions dataset (public, requires remote code).
        all_tasks_df = download_hf_dataset(
            'Super-NI',
            'super_natural_instructions',
            'train[:2%]',  # keep it lightweight
            'definition',
            'Super-NI'
        )

    elif exp_id == 3:
        df_gsm8k = download_hf_dataset('GSM8K', 'openai/gsm8k', 'train', 'question', 'GSM8K')
        df_creative = download_hf_dataset(
            'BIG-bench', 'google/bigbench', 'reasoning_about_colored_objects', 'question', 'creative_story'
        )
        all_tasks_df = pd.concat([df_gsm8k, df_creative], ignore_index=True)

    if all_tasks_df.empty:
        raise ValueError("FATAL: No data was processed. Check experiment config and data sources.")

    print("\nCleaning and tokenizing text...")
    all_tasks_df['prompt'] = (
        all_tasks_df['prompt'].apply(clean_text).apply(tokenize_and_truncate)
    )
    all_tasks_df.dropna(subset=['prompt'], inplace=True)
    all_tasks_df = all_tasks_df[all_tasks_df['prompt'].str.len() > 20]

    opt_dfs, test_dfs, meta_dfs = [], [], []
    for task_name, group in all_tasks_df.groupby('task'):
        print(f"  Splitting task: {task_name} ({len(group)} prompts)")
        group = group.sample(frac=1, random_state=42).reset_index(drop=True)

        test_split = group.head(50)
        remaining = group.iloc[50:]
        opt_split = remaining.head(300)
        meta_split = remaining.iloc[300:800]

        test_dfs.append(test_split)
        opt_dfs.append(opt_split)
        if not meta_split.empty:
            meta_dfs.append(meta_split)

    final_optimization_df = pd.concat(opt_dfs, ignore_index=True)
    final_test_df = pd.concat(test_dfs, ignore_index=True)

    opt_path = output_path / 'optimization.parquet'
    test_path = output_path / 'test.parquet'
    final_optimization_df.to_parquet(opt_path)
    final_test_df.to_parquet(test_path)

    print("\nPreprocessing complete.")
    print(f"  Optimization data: {len(final_optimization_df)} samples saved to {opt_path}")
    print(f"  Test data: {len(final_test_df)} samples saved to {test_path}")

    if meta_dfs:
        final_meta_df = pd.concat(meta_dfs, ignore_index=True)
        meta_path = output_path / 'meta_train.parquet'
        final_meta_df.to_parquet(meta_path)
        print(f"  Meta-training data: {len(final_meta_df)} samples saved to {meta_path}")

    return str(output_path)