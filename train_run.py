"""
Скрипт обучения — зеркало notebooks/train.ipynb.
Все параметры берутся из config.py.

Запуск:
    python3 train_run.py [run_name]

Если run_name не указан, используется RUN_NAME из скрипта.
Результаты пишутся в results/{run_name}/ по схеме из config.py.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score

from config import (
    ROOT, DATASET_ROOT, RESULTS_ROOT, SEED,
    BATCH_SIZE, MAX_EPOCHS, PATIENCE, MIN_DELTA, NUM_WORKERS,
    MODEL_CONFIGS, MODALITIES, CLASS_PALETTE, CLASS_SHORT,
)
from src.utils import set_seed
from src.transforms import get_transforms
from src.data import prepare_loaders
from src.models.resnet import create_resnet18
from src.training import train_one_epoch, validate, EarlyStopping, save_history, step_scheduler

# ── Имя запуска (папка внутри results/) ──────────────────────────────────────
RUN_NAME = sys.argv[1] if len(sys.argv) > 1 else 'run_01'
RUN_DIR  = RESULTS_ROOT / RUN_NAME
PLOTS_DIR = RUN_DIR / 'plots'
RUN_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Device     : {DEVICE}')
print(f'Data       : {DATASET_ROOT}')
print(f'Run        : {RUN_DIR}')
print(f'Epochs     : {MAX_EPOCHS}  |  Patience : {PATIENCE}  |  Batch : {BATCH_SIZE}')
print(f'Workers    : {NUM_WORKERS}')

# Сохраняем конфиг запуска
run_config = {
    'run_name':   RUN_NAME,
    'batch_size': BATCH_SIZE,
    'max_epochs': MAX_EPOCHS,
    'patience':   PATIENCE,
    'min_delta':  MIN_DELTA,
    'seed':       SEED,
    'model_configs': MODEL_CONFIGS,
}
with open(RUN_DIR / 'config.json', 'w', encoding='utf-8') as f:
    json.dump(run_config, f, ensure_ascii=False, indent=2)


def train_model(modality, cfg):
    gen = set_seed(SEED)
    print(f"\n{'='*60}")
    print(f'  Модальность : {modality}')
    print(f"  LR={cfg['lr']:.0e}  WD={cfg['wd']:.0e}  Drop={cfg['dropout']}  Freeze={cfg['freeze']}")
    print(f"  Resize={cfg['resize']}  Aug={cfg['aug']}")
    print(f"{'='*60}")

    train_loader, val_loader, classes = prepare_loaders(
        DATASET_ROOT / modality,
        train_transform=get_transforms(cfg['resize'], cfg['aug'], is_train=True),
        val_transform=get_transforms(cfg['resize'], cfg['aug'], is_train=False),
        batch_size=BATCH_SIZE,
        generator=gen,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == 'cuda'),
    )
    print(f'Классов: {len(classes)} | Трейн: {len(train_loader.dataset)} | Вал: {len(val_loader.dataset)}')

    model     = create_resnet18(len(classes), freeze_mode=cfg['freeze'], dropout_p=cfg['dropout']).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6
    )
    criterion = nn.CrossEntropyLoss()
    es        = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA)

    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, epoch, MAX_EPOCHS)
        val_mets   = validate(model, val_loader, criterion, DEVICE, epoch, MAX_EPOCHS)
        step_scheduler(scheduler, 'plateau', val_mets['f1'])

        if es.step(val_mets['f1'], epoch):
            torch.save(model.state_dict(), RUN_DIR / f'{modality}_best.pth')

        history.append({'epoch': epoch, 'train_loss': train_loss,
                        **{f'val_{k}': v for k, v in val_mets.items()}})
        cur_lr = optimizer.param_groups[0]['lr']
        print(f"Ep {epoch:3d} | TrL {train_loss:.4f} | VL {val_mets['loss']:.4f} | "
              f"F1 {val_mets['f1']:.4f} | Acc {val_mets['acc']:.4f} | lr={cur_lr:.2e} | {es.status}")

        if es.should_stop:
            print(f'\nEarly stop. Лучшая эпоха: {es.best_epoch}  F1={es.best:.4f}')
            break

    save_history(history, RUN_DIR / f'{modality}_history.json')
    torch.cuda.empty_cache()
    return history, classes


all_results = {}
for mod, cfg in MODEL_CONFIGS.items():
    history, classes = train_model(mod, cfg)
    all_results[mod] = {'history': history, 'classes': classes}


# ── Графики: кривые обучения + confusion matrix ───────────────────────────────
def plot_training_curves(mod, hist, save_path):
    epochs = [h['epoch'] for h in hist]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, [h['train_loss'] for h in hist], label='Train')
    axes[0].plot(epochs, [h['val_loss']   for h in hist], label='Val')
    axes[0].set_title(f'{mod} — Loss'); axes[0].set_xlabel('Epoch'); axes[0].legend()

    best_ep = max(hist, key=lambda h: h['val_f1'])
    axes[1].plot(epochs, [h['val_f1'] for h in hist], color='tab:green')
    axes[1].axvline(best_ep['epoch'], color='gray', linestyle='--', alpha=0.6)
    axes[1].set_title(f"{mod} — Val F1 (best={best_ep['val_f1']:.3f})")
    axes[1].set_xlabel('Epoch')

    axes[2].plot(epochs, [h['val_acc']  for h in hist], label='Acc')
    axes[2].plot(epochs, [h['val_prec'] for h in hist], label='Prec')
    axes[2].plot(epochs, [h['val_rec']  for h in hist], label='Rec')
    axes[2].set_title(f'{mod} — Val metrics'); axes[2].set_xlabel('Epoch'); axes[2].legend()

    plt.suptitle(f'Кривые обучения — {mod}', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Сохранено: {save_path}')


def plot_confusion(mod, classes, save_path):
    cfg = MODEL_CONFIGS[mod]
    gen = set_seed(SEED)
    _, val_loader, _ = prepare_loaders(
        DATASET_ROOT / mod,
        train_transform=get_transforms(cfg['resize'], cfg['aug'], is_train=True),
        val_transform=get_transforms(cfg['resize'], cfg['aug'], is_train=False),
        batch_size=BATCH_SIZE, generator=gen,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == 'cuda'),
    )
    model = create_resnet18(len(classes), freeze_mode=cfg['freeze'], dropout_p=cfg['dropout']).to(DEVICE)
    model.load_state_dict(torch.load(RUN_DIR / f'{mod}_best.pth', map_location=DEVICE, weights_only=True))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            _, preds = torch.max(model(inputs.to(DEVICE)), 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    short = [CLASS_SHORT.get(c, c[:8]) for c in classes]
    fig, ax = plt.subplots(figsize=(9, 7))
    disp = ConfusionMatrixDisplay(
        confusion_matrix(all_labels, all_preds, normalize='true'),
        display_labels=short,
    )
    disp.plot(ax=ax, colorbar=True, xticks_rotation=40, values_format='.2f')
    ax.set_title(f'{mod} — Confusion matrix (val, нормировано)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Сохранено: {save_path}')


def plot_per_class_f1(mod, classes, hist, save_path):
    cfg = MODEL_CONFIGS[mod]
    gen = set_seed(SEED)
    _, val_loader, _ = prepare_loaders(
        DATASET_ROOT / mod,
        train_transform=get_transforms(cfg['resize'], cfg['aug'], is_train=True),
        val_transform=get_transforms(cfg['resize'], cfg['aug'], is_train=False),
        batch_size=BATCH_SIZE, generator=gen,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == 'cuda'),
    )
    model = create_resnet18(len(classes), freeze_mode=cfg['freeze'], dropout_p=cfg['dropout']).to(DEVICE)
    model.load_state_dict(torch.load(RUN_DIR / f'{mod}_best.pth', map_location=DEVICE, weights_only=True))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            _, preds = torch.max(model(inputs.to(DEVICE)), 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    f1_per = f1_score(all_labels, all_preds, labels=list(range(len(classes))),
                      average=None, zero_division=0)
    short  = [CLASS_SHORT.get(c, c[:8]) for c in classes]
    colors = [CLASS_PALETTE.get(c, '#94A3B8') for c in classes]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(short, f1_per, color=colors, edgecolor='#444', linewidth=0.6)
    for bar, val in zip(bars, f1_per):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    ax.axhline(f1_per.mean(), color='gray', linestyle='--', linewidth=1,
               label=f'macro avg = {f1_per.mean():.2f}')
    ax.set_ylim(0, 1.05)
    ax.set_title(f'F1 по классам (val) — {mod}')
    ax.tick_params(axis='x', rotation=35)
    ax.legend()
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Сохранено: {save_path}')


for mod, data in all_results.items():
    hist    = data['history']
    classes = data['classes']
    plot_training_curves(mod, hist, PLOTS_DIR / f'curves_{mod}.png')
    plot_confusion(mod, classes,    PLOTS_DIR / f'confusion_{mod}.png')
    plot_per_class_f1(mod, classes, hist, PLOTS_DIR / f'per_class_f1_{mod}.png')


# ── Итоговая таблица ─────────────────────────────────────────────────────────
print('\n' + '='*60)
print(f"{'Мод':6} {'F1':>8} {'Acc':>8} {'Prec':>8} {'Rec':>8}  Эпоха")
print('='*60)
for mod, data in all_results.items():
    best = max(data['history'], key=lambda h: h['val_f1'])
    print(f"{mod:6} {best['val_f1']:8.4f} {best['val_acc']:8.4f} "
          f"{best['val_prec']:8.4f} {best['val_rec']:8.4f}  ep{best['epoch']}")
print('='*60)
print(f'\nРезультаты: {RUN_DIR}')
