import os
import json
from datetime import datetime
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def _select_device() -> torch.device:
    """Select GPU if available, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_model(num_classes: int = 10) -> nn.Module:
    """Returns a ResNet-18 model with the classifier head adapted to *num_classes*."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _loss_fn() -> nn.Module:
    return nn.CrossEntropyLoss()


def _optimizer(model: nn.Module, lr: float) -> optim.Optimizer:
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)


def train(config: Dict):
    """Full training loop; saves the best model to *config['checkpoint_path']*."""
    device = _select_device()

    # ---------------------------------------------------------------------
    # Data ----------------------------------------------------------------
    # ---------------------------------------------------------------------
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    root = os.path.expanduser(config["data_root"])
    train_set = datasets.CIFAR10(root, train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10(root, train=False, download=True, transform=test_tf)

    if config["subset"]:
        # smoke-test mode – use a 5-k sample subset
        train_set, _ = torch.utils.data.random_split(train_set, [5_000, len(train_set) - 5_000],
                                                     generator=torch.Generator().manual_seed(0))
        test_set, _ = torch.utils.data.random_split(test_set, [1_000, len(test_set) - 1_000],
                                                    generator=torch.Generator().manual_seed(0))

    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=config["batch_size"], shuffle=False, num_workers=4, pin_memory=True)

    # ---------------------------------------------------------------------
    # Model ---------------------------------------------------------------
    # ---------------------------------------------------------------------
    model = _build_model(num_classes=10).to(device)
    criterion = _loss_fn()
    optimizer = _optimizer(model, lr=config["learning_rate"])

    best_acc = 0.0
    os.makedirs(os.path.dirname(config["checkpoint_path"]), exist_ok=True)

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        acc = evaluate(model, test_loader, device)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), config["checkpoint_path"])

        print(f"Epoch {epoch:02d}/{config['epochs']}  loss={train_loss:.3f}  acc={acc:.3f}")

    return best_acc


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
    return correct / len(dataloader.dataset)
