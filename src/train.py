import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch.utils.data import DataLoader


class SimpleLanguageModel(torch.nn.Module):
    """A minimal LSTM language-model to keep the example lightweight.

    The implementation purposefully stays *very* small so that a 1-epoch
    smoke-test finishes in <30 s on CPU while still exercising the full
    training/evaluation pipeline.  In the full experiment we merely crank
    up the number of epochs/hidden units in the YAML config – no code
    changes required.
    """

    def __init__(self, vocab_size: int, hidden_size: int = 64):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)
        self.rnn = torch.nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.head = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # shape (B, T)
        h = self.embed(x)
        h, _ = self.rnn(h)
        return self.head(h)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def train_model(train_loader: DataLoader, vocab_size: int, cfg: Dict[str, Any]) -> Tuple[SimpleLanguageModel, Dict[str, float]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleLanguageModel(vocab_size, hidden_size=cfg["model"]["hidden"]).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"])
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    total_loss, n_tokens = 0.0, 0
    for _ in range(cfg["training"]["epochs"]):
        for batch in train_loader:
            optimiser.zero_grad()
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimiser.step()
            total_loss += loss.item() * y.numel()
            n_tokens += y.numel()

    stats = {"train_loss_per_token": total_loss / max(1, n_tokens)}
    return model, stats
