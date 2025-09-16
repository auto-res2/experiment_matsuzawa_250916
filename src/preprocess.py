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


def build_dataloaders(cfg: Dict[str, Any]):
    """Returns train / val dataloaders according to the YAML config."""

    name, split = cfg["data"]["dataset"], cfg["data"].get("split", "train")

    # Use the HF_TOKEN env-var for gated sets (if necessary)
    auth_token = os.getenv("HF_TOKEN")
    ds = datasets.load_dataset(name, split=split, token=auth_token)

    # Bare-bones WordPiece tokeniser from 🤗 – small & fast to load.
    from transformers import AutoTokenizer

    tokenizer_name = cfg["data"].get("tokenizer", "bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    ds = ds.map(functools.partial(_tokenise, tokenizer=tokenizer), batched=True, remove_columns=ds.column_names)
    ds.set_format(type="torch")

    train_loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=True)
    return train_loader, tokenizer.vocab_size
