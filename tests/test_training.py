"""src/training.py:train_task — регрессия на реальный баг, найденный при
сквозной проверке: OneCycleLR падал с ValueError('Expected float between
0 and 1 pct_start') всякий раз, когда --max-epochs меньше config.WARMUP_EPOCHS
(по умолчанию 15) — раньше это никогда не триггерилось, потому что все
исторические прогоны использовали --max-epochs 40/60 или --scheduler plateau."""

import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training import train_task


class _TinyNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, n_classes)

    def forward(self, x):
        return self.fc(self.pool(self.conv(x)).flatten(1))


def _make_loaders(n=32, n_classes=3):
    X = torch.randn(n, 3, 16, 16)
    y = torch.randint(0, n_classes, (n,))
    train_ds = TensorDataset(X, y)
    train_ds.targets = y.tolist()
    val_ds = TensorDataset(X[:8], y[:8])
    return (DataLoader(train_ds, batch_size=8, shuffle=True),
            DataLoader(val_ds, batch_size=8, shuffle=False))


def test_train_task_onecycle_survives_max_epochs_below_warmup():
    """--max-epochs 1 with the default onecycle scheduler and
    warmup_epochs=15 must not raise — this is exactly config.py's default
    (scheduler='onecycle' unless overridden)."""
    train_ldr, val_ldr = _make_loaders()
    model = _TinyNet(3)
    with tempfile.TemporaryDirectory() as tmp:
        history = train_task(
            "ДС", model, train_ldr, val_ldr,
            lr=1e-3, mode="single", device=torch.device("cpu"), run_dir=Path(tmp),
            model_configs={"ДС": {"scheduler": "onecycle", "mix_aug": False, "loss_type": "ce"}},
            max_epochs=1, patience=5, min_delta=0.0, warmup_epochs=15,
        )
    assert len(history) == 1
