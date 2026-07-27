"""
Цикл обучения. Раньше это было продублировано: train_one_epoch/validate/
step_scheduler лежали здесь мёртвым кодом (train.py их не вызывал — только
EarlyStopping и save_history), а реальный, используемый цикл был целиком
инлайнен в train.py. Теперь реальная логика (make_step_fn/run_train_epoch/
run_val_epoch/train_task) живёт здесь; train.py — тонкий CLI поверх неё.
"""

import os
import signal
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from src.augmentation import cutmix_data, cutmix_data_paired
from src.losses import get_criterion, mean_kl_divergence

# ── Мягкая остановка (STOP-файл / SIGUSR1) ───────────────────────────────────
# Ctrl+C (SIGINT) идёт всей группе процессов и убивает DataLoader-воркеров
# раньше, чем main успевает сохранить историю — отсюда второй, "мягкий"
# механизм, который проверяется только между эпохами. Раньше это было
# отдельным proof-of-concept скриптом (test_stop.py), никогда не подключённым
# к train.py — теперь это часть train_task().
STOP_FILE = Path(os.environ.get("ML_STOP_FILE", "/tmp/ml_STOP"))
PID_FILE = Path(os.environ.get("ML_PID_FILE", "/tmp/ml_pid.txt"))


class GracefulStop:
    """Проверяется в начале каждой эпохи. Не трогает DataLoader-воркеров —
    в отличие от Ctrl+C, безопасна для многочасовых overnight-прогонов."""

    def __init__(self) -> None:
        self._requested = False
        self._reason = ""
        signal.signal(signal.SIGUSR1, self._on_signal)

    def _on_signal(self, signum, frame) -> None:
        self.request("SIGUSR1")

    def request(self, reason: str) -> None:
        if not self._requested:
            self._requested = True
            self._reason = reason
            print(f"\n  [graceful-stop] Запрошена остановка ({reason}). "
                  f"Завершаем после текущей эпохи...", flush=True)

    def check(self) -> bool:
        """Вызывать в начале каждой эпохи. Возвращает True, если нужно
        остановиться (уже запрошено, либо только что найден STOP_FILE)."""
        if STOP_FILE.exists():
            STOP_FILE.unlink(missing_ok=True)
            self.request(f"файл {STOP_FILE}")
        return self._requested

    def __enter__(self) -> "GracefulStop":
        PID_FILE.write_text(str(os.getpid()))
        STOP_FILE.unlink(missing_ok=True)
        return self

    def __exit__(self, *exc) -> None:
        PID_FILE.unlink(missing_ok=True)
        STOP_FILE.unlink(missing_ok=True)


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


def save_history(history: List[Dict], path: Path) -> None:
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Forward step (single / dual, CutMix только при training=True)
# ─────────────────────────────────────────────────────────────────────────────

def make_step_fn(model, mode: str, device, use_mix: bool, soft: bool = False):
    if mode == "single":
        _mix = cutmix_data if use_mix else None

        def step(batch, training=True):
            x, labels = batch
            x, labels = x.to(device), labels.to(device)
            if training and _mix:
                x, y_a, y_b, lam = _mix(x, labels)
            else:
                y_a = y_b = labels
                lam = 1.0
            return model(x), y_a, y_b, lam

    elif soft:
        # Dual-stream с мягкими метками. CutMix смешивает распределения линейно.
        _mix = cutmix_data_paired if use_mix else None

        def step(batch, training=True):
            (x_ds, x_uv), soft_labels = batch
            x_ds = x_ds.to(device)
            x_uv = x_uv.to(device)
            soft_labels = soft_labels.to(device)
            if training and _mix:
                x_ds, x_uv, y_a, y_b, lam = _mix(x_ds, x_uv, soft_labels)
                y_mixed = lam * y_a + (1.0 - lam) * y_b
                return model(x_ds, x_uv), y_mixed, y_mixed, 1.0
            return model(x_ds, x_uv), soft_labels, soft_labels, 1.0

    else:
        _mix = cutmix_data_paired if use_mix else None

        def step(batch, training=True):
            (x_ds, x_uv), labels = batch
            x_ds, x_uv, labels = x_ds.to(device), x_uv.to(device), labels.to(device)
            if training and _mix:
                x_ds, x_uv, y_a, y_b, lam = _mix(x_ds, x_uv, labels)
            else:
                y_a = y_b = labels
                lam = 1.0
            return model(x_ds, x_uv), y_a, y_b, lam

    return step


def run_train_epoch(model, loader, step_fn, criterion, optimizer,
                    clip_grad: float, batch_sched, desc: str) -> float:
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc=desc, leave=False):
        outputs, y_a, y_b, lam = step_fn(batch, training=True)
        optimizer.zero_grad()
        loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
        loss.backward()
        if clip_grad > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        if batch_sched:
            batch_sched.step()
        total_loss += loss.item() * y_a.size(0)
    return total_loss / len(loader.dataset)


def run_val_epoch(model, loader, step_fn, criterion, desc: str) -> dict:
    model.eval()
    total_loss, total_kl, all_preds, all_labels = 0.0, 0.0, [], []
    is_soft = False
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            outputs, y_a, _, _ = step_fn(batch, training=False)
            total_loss += criterion(outputs, y_a).item() * y_a.size(0)
            all_preds.extend(torch.max(outputs, 1)[1].cpu().numpy())
            if y_a.dim() == 2:
                is_soft = True
                # KL только для soft — точная метрика качества состава
                total_kl += mean_kl_divergence(outputs, y_a).item() * y_a.size(0)
                all_labels.extend(y_a.argmax(dim=1).cpu().numpy())
            else:
                all_labels.extend(y_a.cpu().numpy())
    n = len(loader.dataset)
    result = {
        "loss": total_loss / n,
        "f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "acc": accuracy_score(all_labels, all_preds),
        "prec": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "rec": recall_score(all_labels, all_preds, average="macro", zero_division=0),
    }
    if is_soft:
        result["kl"] = total_kl / n
    return result


def collect_preds(model, val_loader, step_fn):
    import numpy as np

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            outputs, y_a, _, _ = step_fn(batch, training=False)
            all_preds.extend(torch.max(outputs, 1)[1].cpu().numpy())
            if y_a.dim() == 2:
                all_labels.extend(y_a.argmax(dim=1).cpu().numpy())
            else:
                all_labels.extend(y_a.cpu().numpy())
    return np.array(all_labels), np.array(all_preds)


# ─────────────────────────────────────────────────────────────────────────────
# Цикл обучения одной задачи
# ─────────────────────────────────────────────────────────────────────────────

def train_task(name: str, model, train_loader, val_loader,
               lr: float, mode: str, device, run_dir: Path,
               model_configs: dict, max_epochs: int, patience: int, min_delta: float,
               warmup_epochs: int, soft: bool = False) -> List[Dict]:
    """Обучает модель, сохраняет лучший checkpoint. Возвращает history.

    Две независимые точки остановки:
      - Ctrl+C (KeyboardInterrupt) — немедленно, может оборвать DataLoader-воркеров.
      - STOP_FILE/SIGUSR1 (GracefulStop) — только между эпохами, безопасно для
        многочасовых overnight-прогонов (см. модуль docstring и scripts/run_batch.py).
    """
    cfg = model_configs.get(name, {})
    wd = cfg.get("wd", 1e-4)
    clip_grad = cfg.get("clip_grad", 1.0)
    use_mix = cfg.get("mix_aug", True)
    focal_g = cfg.get("focal_gamma", 2.0)

    if soft:
        criterion = get_criterion("soft_ce")
    else:
        counts = torch.bincount(torch.tensor(train_loader.dataset.targets))
        class_weights = (counts.sum() / (len(counts) * counts.float())).to(device)
        criterion = get_criterion(cfg.get("loss_type", "focal"), class_weights, focal_gamma=focal_g)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched_type = cfg.get("scheduler", "onecycle")
    if sched_type == "plateau":
        epoch_sched = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6,
        )
        batch_sched = None
    else:
        epoch_sched = None
        # pct_start должен быть в (0, 1) — warmup_epochs (по умолчанию 15, см.
        # config.WARMUP_EPOCHS) не масштабируется автоматически при
        # --max-epochs < warmup_epochs (например, smoke-тест на 1 эпоху).
        # Раньше это падало с ValueError из OneCycleLR прямо в середине
        # прогона; теперь warmup доля просто ограничена разумным диапазоном.
        pct_start = max(0.05, min(0.9, warmup_epochs / max_epochs))
        batch_sched = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, epochs=max_epochs, steps_per_epoch=len(train_loader),
            pct_start=pct_start, div_factor=10.0, final_div_factor=1e4,
        )
    es = EarlyStopping(patience=patience, min_delta=min_delta)
    step_fn = make_step_fn(model, mode, device, use_mix, soft=soft)
    ckpt = run_dir / f"{name}_best.pth"

    print(f"\n{'=' * 60}")
    print(f"  {name}  |  LR={lr:.0e}  WD={wd:.0e}  Batch={train_loader.batch_size}  Sched={sched_type}")
    print(f"  Трейн: {len(train_loader.dataset):,}  |  Вал: {len(val_loader.dataset):,}")
    print(f"{'=' * 60}")

    history: List[Dict] = []
    with GracefulStop() as stop:
        try:
            for epoch in range(1, max_epochs + 1):
                if stop.check():
                    print(f"\n  [graceful-stop] Остановлено перед эпохой {epoch}. "
                          f"Лучшая: ep{es.best_epoch}  F1={es.best:.4f}")
                    break

                train_loss = run_train_epoch(
                    model, train_loader, step_fn, criterion, optimizer,
                    clip_grad, batch_sched, f"Train {epoch}/{max_epochs}",
                )
                val_mets = run_val_epoch(
                    model, val_loader, step_fn, criterion, f"Val   {epoch}/{max_epochs}",
                )

                # Метрика для early stopping:
                # soft режим → -KL (меньше KL = лучше, инвертируем для "больше = лучше")
                # hard режим → F1 (больше = лучше)
                es_metric = -val_mets["kl"] if (soft and "kl" in val_mets) else val_mets["f1"]

                if epoch_sched is not None:
                    epoch_sched.step(es_metric)

                if es.step(es_metric, epoch):
                    torch.save(model.state_dict(), ckpt)

                history.append({"epoch": epoch, "train_loss": train_loss,
                                **{f"val_{k}": v for k, v in val_mets.items()}})
                cur_lr = optimizer.param_groups[0]["lr"]
                kl_str = f" | KL {val_mets['kl']:.3f}" if "kl" in val_mets else ""
                print(f"Ep {epoch:3d} | TrL {train_loss:.4f} | VL {val_mets['loss']:.4f} | "
                      f"F1 {val_mets['f1']:.4f} | Acc {val_mets['acc']:.4f}{kl_str} | "
                      f"lr={cur_lr:.2e} | {es.status}")

                if es.should_stop:
                    print(f"\n  [early stop] Лучшая эпоха: {es.best_epoch}  F1={es.best:.4f}")
                    break

        except KeyboardInterrupt:
            print(f"\n  [Ctrl+C] Остановлено на эпохе {len(history)}. "
                  f"Лучшая: ep{es.best_epoch}  F1={es.best:.4f}")

    save_history(history, run_dir / f"{name}_history.json")
    torch.cuda.empty_cache()
    return history
