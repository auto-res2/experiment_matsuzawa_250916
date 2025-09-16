"""Data-loading and tokenisation helpers.

The module purposefully relies only on `datasets` + `torch` so that the
resulting project stays lightweight and easily installable inside a fresh
CI sandbox.
"""

import functools
import os
from typing import Any, Dict

import datasets
import torch
from torch.utils.data import DataLoader


def _tokenise(example: Dict[str, Any], tokenizer) -> Dict[str, Any]:
    ids = tokenizer(example["text"], truncation=True, padding="max_length")
    example["input_ids"] = ids["input_ids"]
    example["labels"] = ids["input_ids"].copy()
    return example


def _load_hf_dataset(cfg: Dict[str, Any]):
    """Robust wrapper around `datasets.load_dataset` handling config names.

    The YAML may specify either
      data:
        dataset: wikitext
        split: wikitext-2-raw-v1   # (legacy key – actually the *config* name)

    or the newer explicit form
      data:
        dataset: wikitext
        config: wikitext-2-raw-v1  # ✅ clear intent
        hf_split: train            # optional, defaults to "train"
    """
    name = cfg["data"]["dataset"]
    config_name = cfg["data"].get("config")
    # `hf_split` == HuggingFace split name (train/validation/test)
    hf_split = cfg["data"].get("hf_split", "train")

    # Use the HF_TOKEN env-var for gated sets (if necessary)
    auth_token = os.getenv("HF_TOKEN")

    # ------------------------------------------------------------------
    # Fast-path – config explicitly provided in YAML
    # ------------------------------------------------------------------
    if config_name is not None:
        return datasets.load_dataset(name, config_name, split=hf_split, token=auth_token)

    # ------------------------------------------------------------------
    # Legacy path – try loading without a config; if the dataset *requires*
    # a config (e.g. wikitext) we inspect the raised ValueError and retry
    # with cfg["data"]["split"] being treated as the config name.
    # ------------------------------------------------------------------
    try:
        return datasets.load_dataset(name, split=hf_split, token=auth_token)
    except ValueError as err:
        if "Config name is missing" in str(err):
            # Fall back to legacy behaviour where `split` actually holds the
            # HF *config* (e.g. "wikitext-2-raw-v1")
            legacy_cfg_name = cfg["data"].get("split")
            if legacy_cfg_name is None:
                raise  # re-throw – nothing we can do.
            return datasets.load_dataset(name, legacy_cfg_name, split=hf_split, token=auth_token)
        raise  # genuine error – bubble up.


def build_dataloaders(cfg: Dict[str, Any]):
    """Returns a single train DataLoader according to the YAML config."""

    ds = _load_hf_dataset(cfg)

    # Bare-bones WordPiece tokeniser from 🤗 – small & fast to load.
    from transformers import AutoTokenizer

    tokenizer_name = cfg["data"].get("tokenizer", "bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    ds = ds.map(functools.partial(_tokenise, tokenizer=tokenizer), batched=True, remove_columns=ds.column_names)
    ds.set_format(type="torch")

    train_loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=True)
    return train_loader, tokenizer.vocab_size
