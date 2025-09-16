import json
import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from .train import evaluate, _select_device


def run_evaluation(checkpoint_path: str, batch_size: int, results_dir: str):
    device = _select_device()

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    test_set = datasets.CIFAR10(os.path.expanduser("~/.cache/data"), train=False, download=True, transform=tf)
    dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)

    acc = evaluate(model, dataloader, device)

    os.makedirs(results_dir, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(results_dir, f"results_{stamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"accuracy": acc}, f, indent=2)
    print(json.dumps({"accuracy": acc}, indent=2))
