import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def evaluate_model(model: torch.nn.Module, data_loader: DataLoader, cfg: Dict[str, Any]) -> Dict[str, float]:
    device = next(model.parameters()).device
    model.eval()
    nll, n_tokens = 0.0, 0
    with torch.no_grad():
        for batch in data_loader:
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)
            logits = model(x)
            log_prob = F.log_softmax(logits, dim=-1)
            nll -= (log_prob.gather(-1, y.unsqueeze(-1)).squeeze(-1)).sum().item()
            n_tokens += y.numel()

    ppl = torch.exp(torch.tensor(nll / max(n_tokens, 1))).item()
    return {"neg_log_likelihood": nll / max(1, n_tokens), "perplexity": ppl}
