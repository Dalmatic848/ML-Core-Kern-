"""
Единый скрипт обучения — ResNet18/50, single/dual stream.

Запуск напрямую:
    python3 train.py run_v5
    python3 train.py run_v5         --mode dual
    python3 train.py run_v5         --modality ДС        # только одна модальность
    python3 train.py run_rn50_v1    --arch resnet50
    python3 train.py run_rn50_dual  --arch resnet50 --mode dual --warm-start results/run_rn50_v1

Рекомендуется запускать через run.sh — он разбивает ДС и УФ на отдельные процессы,
так что Ctrl+C завершает только текущий шаг и пайплайн продолжается.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import json

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets as tv_datasets

from config import (
    BATCH_SIZE,
    CLASS_PALETTE,
    CLASS_SHORT,
    DATASET_ROOT,
    DS_MEAN,
    DS_STD,
    DUAL_BATCH_SIZE,
    DUAL_LR,
    MAX_EPOCHS,
    MIN_DELTA,
    MODALITIES,
    MODEL_CONFIGS,
    NUM_WORKERS,
    PATIENCE,
    RESULTS_ROOT,
    RN50_BATCH_SIZE,
    RN50_DUAL_BATCH_SIZE,
    RN50_LR,
    SEED,
    UV_MEAN,
    UV_STD,
    WARMUP_EPOCHS,
)
from src.data import PairedDataset, SoftLabelPairedDataset, prepare_loaders, prepare_paired_loaders
from src.models.registry import build_model
from src.training import collect_preds, make_step_fn, train_task
from src.transforms import get_transforms
from src.utils import set_seed


def _make_arch_entry(arch: str, mode: str, batch_size: int, lr: float) -> dict:
    """Создаёт запись реестра через src.models.registry.build_model — единая
    точка arch-dispatch, используемая и в train.py, и в visualize_*.py."""
    return dict(model_fn=lambda n, **kw: build_model(arch, mode, n, **kw),
                batch_size=batch_size, lr=lr)


# ─────────────────────────────────────────────────────────────────────────────
# Реестр архитектур — batch_size подобраны под 8GB VRAM
# ─────────────────────────────────────────────────────────────────────────────
ARCH_REGISTRY = {
    ('resnet18', 'single'): _make_arch_entry('resnet18', 'single', BATCH_SIZE,      3e-4),
    ('resnet18', 'dual'):   _make_arch_entry('resnet18', 'dual',   DUAL_BATCH_SIZE, DUAL_LR),
    ('resnet50', 'single'): _make_arch_entry('resnet50', 'single', RN50_BATCH_SIZE,      RN50_LR),
    ('resnet50', 'dual'):   _make_arch_entry('resnet50', 'dual',   RN50_DUAL_BATCH_SIZE, RN50_LR),
    # EfficientNet-B3 — batch=8 dual (24.5M params × 2, RTX 3050 8GiB)
    ('efficientnet_b3', 'single'): _make_arch_entry('efficientnet_b3', 'single', 24, 2e-4),
    ('efficientnet_b3', 'dual'):   _make_arch_entry('efficientnet_b3', 'dual',    8, 1e-4),
    # ConvNeXt-Tiny — batch=8 dual (56.8M params, самая тяжёлая)
    ('convnext_tiny', 'single'): _make_arch_entry('convnext_tiny', 'single', 16, 2e-4),
    ('convnext_tiny', 'dual'):   _make_arch_entry('convnext_tiny', 'dual',    8, 1e-4),
}


# ─────────────────────────────────────────────────────────────────────────────
# Загрузчики для обучения (с воркерами)
# ─────────────────────────────────────────────────────────────────────────────
def make_single_loaders(modality: str, batch_size: int, gen=None):
    cfg = MODEL_CONFIGS[modality]
    mean, std = (DS_MEAN, DS_STD) if modality == 'ДС' else (UV_MEAN, UV_STD)
    return prepare_loaders(
        DATASET_ROOT / modality,
        train_transform=get_transforms(cfg['resize'], cfg['aug'], True,  mean, std),
        val_transform  =get_transforms(cfg['resize'], 'none',     False, mean, std),
        batch_size=batch_size, generator=gen,
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
        use_sampler=cfg.get('use_sampler', False),
    )


def make_dual_loaders(batch_size: int, gen=None):
    cfg_ds, cfg_uv = MODEL_CONFIGS['ДС'], MODEL_CONFIGS['УФ']
    return prepare_paired_loaders(
        DATASET_ROOT,
        ds_train_tfm=get_transforms(cfg_ds['resize'], cfg_ds['aug'], True,  DS_MEAN, DS_STD),
        ds_val_tfm  =get_transforms(cfg_ds['resize'], 'none',        False, DS_MEAN, DS_STD),
        uv_train_tfm=get_transforms(cfg_uv['resize'], cfg_uv['aug'], True,  UV_MEAN, UV_STD),
        uv_val_tfm  =get_transforms(cfg_uv['resize'], 'none',        False, UV_MEAN, UV_STD),
        batch_size=batch_size, generator=gen,
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
    )


def make_soft_dual_loaders(soft_labels: dict, batch_size: int, gen=None):
    """Загрузчики для soft labels (dual режим). soft_labels: {class_name: [p0..p4]}"""
    cfg_ds, cfg_uv = MODEL_CONFIGS['ДС'], MODEL_CONFIGS['УФ']

    def _make(split, is_train):
        ds = SoftLabelPairedDataset(
            DATASET_ROOT / 'ДС' / split,
            DATASET_ROOT / 'УФ' / split,
            get_transforms(cfg_ds['resize'], cfg_ds['aug'] if is_train else 'none', is_train, DS_MEAN, DS_STD),
            get_transforms(cfg_uv['resize'], cfg_uv['aug'] if is_train else 'none', is_train, UV_MEAN, UV_STD),
            soft_labels,
        )
        return ds

    train_ds = _make('train', True)
    val_ds   = _make('val',   False)
    classes  = list(train_ds.soft_labels.keys())  # оригинальные классы для информации

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin,
                              drop_last=True, persistent_workers=NUM_WORKERS > 0,
                              generator=gen)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin,
                              persistent_workers=NUM_WORKERS > 0)
    return train_loader, val_loader, classes


# ─────────────────────────────────────────────────────────────────────────────
# Загрузчики для inference после обучения (num_workers=0)
# Работают даже если обучение было прервано Ctrl+C и воркеры убиты.
# ─────────────────────────────────────────────────────────────────────────────
def make_single_val_loader(modality: str, batch_size: int):
    cfg = MODEL_CONFIGS[modality]
    mean, std = (DS_MEAN, DS_STD) if modality == 'ДС' else (UV_MEAN, UV_STD)
    ds = tv_datasets.ImageFolder(
        DATASET_ROOT / modality / 'val',
        transform=get_transforms(cfg['resize'], 'none', False, mean, std),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0), ds.classes


def make_dual_val_loader(batch_size: int):
    cfg_ds, cfg_uv = MODEL_CONFIGS['ДС'], MODEL_CONFIGS['УФ']
    ds = PairedDataset(
        DATASET_ROOT / 'ДС' / 'val',
        DATASET_ROOT / 'УФ' / 'val',
        get_transforms(cfg_ds['resize'], 'none', False, DS_MEAN, DS_STD),
        get_transforms(cfg_uv['resize'], 'none', False, UV_MEAN, UV_STD),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0), ds.classes




# ─────────────────────────────────────────────────────────────────────────────
# Графики
# ─────────────────────────────────────────────────────────────────────────────
def _plot_curves(name: str, hist: list, arch_label: str, save_path: Path):
    if not hist:
        return
    epochs = [h['epoch'] for h in hist]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, [h['train_loss'] for h in hist], label='Train')
    axes[0].plot(epochs, [h['val_loss']   for h in hist], label='Val')
    axes[0].set_title(f'{name} — Loss'); axes[0].legend()
    best = max(hist, key=lambda h: h['val_f1'])
    axes[1].plot(epochs, [h['val_f1'] for h in hist], color='tab:green')
    axes[1].axvline(best['epoch'], color='gray', linestyle='--', alpha=0.6)
    axes[1].set_title(f"{name} — Val F1 (best={best['val_f1']:.3f})")
    axes[2].plot(epochs, [h['val_acc']  for h in hist], label='Acc')
    axes[2].plot(epochs, [h['val_prec'] for h in hist], label='Prec')
    axes[2].plot(epochs, [h['val_rec']  for h in hist], label='Rec')
    axes[2].set_title(f'{name} — Val metrics'); axes[2].legend()
    for ax in axes:
        ax.set_xlabel('Epoch')
    plt.suptitle(f'Кривые обучения — {name} ({arch_label})', fontsize=12)
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  {save_path.name}')


def _plot_confusion(labels, preds, classes: list, title: str, save_path: Path):
    short = [CLASS_SHORT.get(c, c[:8]) for c in classes]
    fig, ax = plt.subplots(figsize=(9, 7))
    ConfusionMatrixDisplay(
        confusion_matrix(labels, preds, normalize='true'), display_labels=short
    ).plot(ax=ax, colorbar=True, xticks_rotation=40, values_format='.2f')
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  {save_path.name}')


def _plot_per_class_f1(labels, preds, classes: list, title: str, save_path: Path):
    f1_per = f1_score(labels, preds, labels=list(range(len(classes))), average=None, zero_division=0)
    short  = [CLASS_SHORT.get(c, c[:8]) for c in classes]
    colors = [CLASS_PALETTE.get(c, '#94A3B8') for c in classes]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(short, f1_per, color=colors, edgecolor='#444', linewidth=0.6)
    for bar, v in zip(bars, f1_per):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    ax.axhline(f1_per.mean(), color='gray', linestyle='--', linewidth=1,
               label=f'macro = {f1_per.mean():.2f}')
    ax.set_ylim(0, 1.05); ax.tick_params(axis='x', rotation=35); ax.legend()
    ax.set_title(title)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  {save_path.name}')


def _plot_summary(results: dict, run_name: str, arch_label: str,
                  plots_dir: Path, run_dir: Path):
    rows, summary = [], []
    for name, data in results.items():
        best = max(data['history'], key=lambda h: h['val_f1'])
        rows.append([name, f"{best['val_f1']:.4f}", f"{best['val_acc']:.4f}",
                     f"{best['val_prec']:.4f}", f"{best['val_rec']:.4f}", f"ep {best['epoch']}"])
        summary.append({'name': name, **best})

    fig, ax = plt.subplots(figsize=(11, max(2.5, 1.2 * len(rows))))
    ax.axis('off')
    tbl = ax.table(
        cellText=rows,
        colLabels=['Модель', 'F1 macro', 'Accuracy', 'Precision', 'Recall', 'Лучшая эпоха'],
        cellLoc='center', loc='center',
    )
    tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1.0, 2.0)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#1E3A5F'); cell.set_text_props(color='white', fontweight='bold')
        elif r % 2:
            cell.set_facecolor('#EFF6FF')
        else:
            cell.set_facecolor('#DBEAFE')
        cell.set_edgecolor('#93C5FD')
    ax.set_title(f'Итоговые метрики — {run_name} ({arch_label})', fontsize=13, pad=16, fontweight='bold')
    plt.tight_layout()
    save_path = plots_dir / 'metrics_summary.png'
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  {save_path.name}')

    with open(run_dir / 'metrics_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def generate_plots(name: str, model, history: list, classes: list,
                   batch_size: int, mode: str, device, arch_label: str,
                   plots_dir: Path, run_dir: Path):
    """Генерирует все графики для одной задачи. Использует num_workers=0 —
    работает даже если обучение было прервано Ctrl+C."""
    ckpt = run_dir / f'{name}_best.pth'
    if not ckpt.exists() or not history:
        print(f'  [SKIP] {name}: нет checkpoint или истории')
        return None, None

    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))

    if mode == 'single':
        val_ldr, classes = make_single_val_loader(name, batch_size)
    else:
        val_ldr, classes = make_dual_val_loader(batch_size)

    step_fn = make_step_fn(model, mode, device, use_mix=False)
    labels, preds = collect_preds(model, val_ldr, step_fn)

    print(f'\nГрафики для {name}:')
    _plot_curves(name, history, arch_label, plots_dir / f'curves_{name}.png')
    _plot_confusion(labels, preds, classes,
                    f'{name} ({arch_label}) — Confusion matrix (val)',
                    plots_dir / f'confusion_{name}.png')
    _plot_per_class_f1(labels, preds, classes,
                       f'F1 по классам (val) — {name} ({arch_label})',
                       plots_dir / f'per_class_f1_{name}.png')
    return labels, preds, classes


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description='Единый скрипт обучения ML-Core-Kern-',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('run_name',
                   help='Имя папки в results/ (например: run_v5)')
    p.add_argument('--arch', default='resnet18',
                   choices=sorted({k[0] for k in ARCH_REGISTRY}),
                   help='Backbone: resnet18 | resnet50 | efficientnet_b3 | convnext_tiny (default: resnet18)')
    p.add_argument('--mode', default='single', choices=['single', 'dual'],
                   help='single — ДС+УФ независимо; dual — мультимодальный (default: single)')
    p.add_argument('--modality', default=None, choices=MODALITIES,
                   help='Обучать только одну модальность (только для --mode single)')
    p.add_argument('--warm-start', default=None, metavar='PATH',
                   help='Папка с ДС_best.pth и УФ_best.pth для тёплого старта (только --mode dual)')
    p.add_argument('--dataset-root', default=None, metavar='PATH',
                   help='Переопределить DATASET_ROOT из config.py (для экспериментальных датасетов)')
    p.add_argument('--soft-labels', action='store_true',
                   help='Использовать мягкие метки из dataset-root/soft_labels.json (только --mode dual)')
    p.add_argument('--scheduler', default=None, choices=['onecycle', 'plateau'],
                   help='Переопределить планировщик LR (default: из MODEL_CONFIGS или onecycle)')
    p.add_argument('--max-epochs', type=int, default=None,
                   help=f'Переопределить MAX_EPOCHS (default: {MAX_EPOCHS} из config.py)')
    p.add_argument('--patience', type=int, default=None,
                   help=f'Переопределить PATIENCE early stopping (default: {PATIENCE})')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # Опциональное переопределение датасета (для экспериментов)
    if args.dataset_root:
        global DATASET_ROOT, DS_MEAN, DS_STD, UV_MEAN, UV_STD
        DATASET_ROOT = Path(args.dataset_root)
        stats_file = DATASET_ROOT / 'normalization_stats.json'
        if stats_file.exists():
            with open(stats_file) as _f:
                _s = json.load(_f)
            DS_MEAN = _s['ДС']['mean']
            DS_STD  = _s['ДС']['std']
            UV_MEAN = _s['УФ']['mean']
            UV_STD  = _s['УФ']['std']
            print(f'Dataset : {DATASET_ROOT}  (нормализация из {stats_file.name})')
        else:
            print(f'Dataset : {DATASET_ROOT}  (normalization_stats.json не найден, используем config)')

    reg  = ARCH_REGISTRY[(args.arch, args.mode)]
    arch_label = f'{args.arch.capitalize()} {args.mode.capitalize()}'

    # Переопределение гиперпараметров через CLI
    if args.max_epochs:
        global MAX_EPOCHS
        MAX_EPOCHS = args.max_epochs
    if args.patience:
        global PATIENCE
        PATIENCE = args.patience

    if args.scheduler:
        for k in MODEL_CONFIGS:
            MODEL_CONFIGS[k]['scheduler'] = args.scheduler
        # Для dual: вшиваем в специальный ключ
        MODEL_CONFIGS.setdefault('dual', {})['scheduler'] = args.scheduler

    run_dir   = RESULTS_ROOT / args.run_name
    plots_dir = run_dir / 'plots'
    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {DEVICE}  |  Arch: {arch_label}  |  Run: {run_dir}')
    print(f'Batch  : {reg["batch_size"]}  |  Epochs: {MAX_EPOCHS}  |  Patience: {PATIENCE}')
    print('Ctrl+C — остановить текущий шаг (графики всё равно будут построены)\n')

    all_results: dict = {}

    # ── Single ────────────────────────────────────────────────────────────────
    if args.mode == 'single':
        mods = [args.modality] if args.modality else MODALITIES
        for mod in mods:
            gen = set_seed(SEED)
            train_ldr, val_ldr, classes = make_single_loaders(mod, reg['batch_size'], gen)
            model = reg['model_fn'](len(classes), freeze_mode='none', dropout_p=0.5).to(DEVICE)

            history = train_task(
                mod, model, train_ldr, val_ldr,
                lr=MODEL_CONFIGS[mod].get('lr', reg['lr']),
                mode='single', device=DEVICE, run_dir=run_dir,
                model_configs=MODEL_CONFIGS, max_epochs=MAX_EPOCHS,
                patience=PATIENCE, min_delta=MIN_DELTA, warmup_epochs=WARMUP_EPOCHS,
            )

            # Графики через свежий загрузчик (num_workers=0) — работает после Ctrl+C
            result = generate_plots(mod, model, history, classes,
                                    reg['batch_size'], 'single', DEVICE,
                                    arch_label, plots_dir, run_dir)
            if result is not None:
                labels, preds, classes = result
                all_results[mod] = {'history': history, 'classes': classes,
                                    'labels': labels, 'preds': preds}

    # ── Dual ──────────────────────────────────────────────────────────────────
    else:
        gen = set_seed(SEED)

        if args.soft_labels:
            # Режим мягких меток: num_classes = число базовых компонент (5)
            sl_path = DATASET_ROOT / 'soft_labels.json'
            bc_path = DATASET_ROOT / 'base_classes.json'
            if not sl_path.exists():
                raise FileNotFoundError(f'Не найден {sl_path}. Создайте датасет: prepare_dataset.py --variant soft')
            with open(sl_path, encoding='utf-8') as _f:
                soft_labels_map = json.load(_f)
            with open(bc_path, encoding='utf-8') as _f:
                base_classes = json.load(_f)
            print(f'Soft labels: {len(soft_labels_map)} исходных классов → {len(base_classes)} базовых компонент')
            print(f'Базовые: {base_classes}')
            train_ldr, val_ldr, _ = make_soft_dual_loaders(soft_labels_map, reg['batch_size'], gen)
            num_classes = len(base_classes)
            classes = base_classes
        else:
            train_ldr, val_ldr, classes = make_dual_loaders(reg['batch_size'], gen)
            num_classes = len(classes)

        model = reg['model_fn'](num_classes, freeze_mode='none', dropout_p=0.5).to(DEVICE)

        if args.warm_start:
            ws = Path(args.warm_start)
            print(f'Тёплый старт из: {ws}')
            model.load_from_single_checkpoints(
                ds_ckpt_path=ws / 'ДС_best.pth',
                uv_ckpt_path=ws / 'УФ_best.pth',
                device=DEVICE,
            )

        history = train_task(
            'dual', model, train_ldr, val_ldr,
            lr=reg['lr'], mode='dual', device=DEVICE, run_dir=run_dir,
            soft=args.soft_labels,
            model_configs=MODEL_CONFIGS, max_epochs=MAX_EPOCHS,
            patience=PATIENCE, min_delta=MIN_DELTA, warmup_epochs=WARMUP_EPOCHS,
        )

        result = generate_plots('dual', model, history, classes,
                                reg['batch_size'], 'dual', DEVICE,
                                arch_label, plots_dir, run_dir)
        if result is not None:
            labels, preds, classes = result
            all_results['dual'] = {'history': history, 'classes': classes,
                                   'labels': labels, 'preds': preds}

    # ── config.json ───────────────────────────────────────────────────────────
    with open(run_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump({
            'run_name':      args.run_name,
            'arch':          args.arch if args.mode == 'single' else f'dual_{args.arch}',
            'mode':          args.mode,
            'batch_size':    reg['batch_size'],
            'max_epochs':    MAX_EPOCHS,
            'patience':      PATIENCE,
            'min_delta':     MIN_DELTA,
            'seed':          SEED,
            'warm_start':    args.warm_start,
            'model_configs': MODEL_CONFIGS,
        }, f, ensure_ascii=False, indent=2)

    if not all_results:
        print('Нет результатов для сводной таблицы.')
        return

    _plot_summary(all_results, args.run_name, arch_label, plots_dir, run_dir)

    print('\n' + '='*60)
    print(f"{'Модель':12} {'F1':>8} {'Acc':>8} {'Prec':>8} {'Rec':>8}  Эпоха")
    print('='*60)
    for name, data in all_results.items():
        best = max(data['history'], key=lambda h: h['val_f1'])
        print(f"{name:12} {best['val_f1']:8.4f} {best['val_acc']:8.4f} "
              f"{best['val_prec']:8.4f} {best['val_rec']:8.4f}  ep{best['epoch']}")
    print('='*60)
    print(f'\nРезультаты: {run_dir}')


if __name__ == '__main__':
    main()
