import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.005):
        self.patience = patience
        self.min_delta = min_delta
        self.best = 0.0
        self.best_epoch = 0
        self._counter = 0

    def step(self, metric: float, epoch: int) -> bool:
        """Возвращает True если метрика улучшилась."""
        if metric > self.best + self.min_delta:
            self.best = metric
            self.best_epoch = epoch
            self._counter = 0
            return True
        self._counter += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self._counter >= self.patience

    @property
    def status(self) -> str:
        return "BEST" if self._counter == 0 else f"wait {self._counter}/{self.patience}"


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    mix_fn: Optional[Callable] = None,
    clip_grad: float = 0.0,
    batch_scheduler=None,
) -> float:
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc=f"Train {epoch}/{total_epochs}", leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        if mix_fn is not None:
            inputs, y_a, y_b, lam = mix_fn(inputs, labels)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
        else:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        loss.backward()
        if clip_grad > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        if batch_scheduler is not None:
            batch_scheduler.step()
        running_loss += loss.item() * inputs.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return running_loss / len(loader.dataset)


def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc=f"Val   {epoch}/{total_epochs}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            running_loss += criterion(outputs, labels).item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    val_loss = running_loss / len(loader.dataset)
    return {
        "loss": val_loss,
        "f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "acc": accuracy_score(all_labels, all_preds),
        "prec": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "rec": recall_score(all_labels, all_preds, average="macro", zero_division=0),
    }


def step_scheduler(scheduler, sched_type: str, metric: Optional[float] = None) -> None:
    """Единая точка вызова scheduler.step() с учётом типа планировщика."""
    if sched_type == "plateau":
        scheduler.step(metric)
    else:
        scheduler.step()


def save_history(history: List[Dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
