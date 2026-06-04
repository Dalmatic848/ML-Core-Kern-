"""
Визуализация результатов: фото керна + метрики на val/test.

Запуск:
    python3 visualize_results.py run_v4
    python3 visualize_results.py run_v4 --split val
    python3 visualize_results.py run_dual_v1 --dual
    python3 visualize_results.py run_v4 --well Харасавэйск_1700 --depth 1718.70
    python3 visualize_results.py run_v4 --wells 3 --no-stats
    python3 visualize_results.py run_v4 --no-photos
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    f1_score, precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from torchvision import datasets as tv_datasets

import config as _cfg
from src.models.resnet import create_resnet18, create_resnet50
from src.models.dual_resnet import (
    create_dual_resnet18, DualStreamResNet18,
    create_dual_resnet50, DualStreamResNet50,
)
from src.transforms import get_transforms
from src.utils import set_seed


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_DEPTH_RE = re.compile(r'_([\d]+[.,][\d]+)\s*-\s*([\d]+[.,][\d]+)$')


# ─────────────────────────────────────────────────────────────────────────────
# Авто-определение num_classes из checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def _num_classes_from_state(state: dict, dual: bool = False) -> int:
    key = 'head.4.weight' if dual else 'fc.1.weight'
    if key in state:
        return state[key].shape[0]
    # Fallback: ищем любой ключ с 'weight' в fc или head
    for k, v in state.items():
        if ('fc' in k or 'head' in k) and k.endswith('.weight') and v.dim() == 2:
            # 512=single RN18, 1024=dual RN18 fusion, 2048=single RN50, 4096=dual RN50 fusion
            if v.shape[1] in (512, 1024, 2048, 4096):
                return v.shape[0]
    raise ValueError(f'Не удалось определить num_classes из checkpoint. Ключи: {list(state)[:10]}')


def _read_arch(run_dir: Path) -> str:
    """Читает архитектуру из config.json; возвращает 'resnet18' если не указана."""
    cfg_path = run_dir / 'config.json'
    if cfg_path.exists():
        with open(cfg_path, encoding='utf-8') as f:
            return json.load(f).get('arch', 'resnet18')
    return 'resnet18'


def _load_single(pth: Path, modality: str, arch: str = 'resnet18') -> torch.nn.Module:
    state = torch.load(pth, map_location=DEVICE, weights_only=True)
    state = {k.replace('module.', '').replace('model.', ''): v for k, v in state.items()}
    n   = _num_classes_from_state(state, dual=False)
    cfg = _cfg.MODEL_CONFIGS[modality]
    if arch == 'resnet50':
        model = create_resnet50(n, freeze_mode=cfg['freeze'], dropout_p=cfg['dropout'])
    else:
        model = create_resnet18(n, freeze_mode=cfg['freeze'], dropout_p=cfg['dropout'])
    model.load_state_dict(state, strict=True)
    return model.to(DEVICE).eval()


def _load_dual(pth: Path, arch: str = 'resnet18') -> torch.nn.Module:
    state = torch.load(pth, map_location=DEVICE, weights_only=True)
    n = _num_classes_from_state(state, dual=True)
    if arch in ('dual_resnet50', 'resnet50'):
        model = create_dual_resnet50(n, dropout_p=0.5)
    else:
        model = create_dual_resnet18(n, dropout_p=0.5)
    model.load_state_dict(state, strict=True)
    return model.to(DEVICE).eval()


def _classes_from_checkpoint(state: dict, dual: bool) -> int:
    return _num_classes_from_state(state, dual)


# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции
# ─────────────────────────────────────────────────────────────────────────────

def _parse_depth(fname: str) -> Optional[tuple]:
    m = _DEPTH_RE.search(fname)
    if not m:
        return None
    return float(m.group(1).replace(',', '.')), float(m.group(2).replace(',', '.'))


def _find_photos(well_name: str, target_depth=None):
    well_dir = _cfg.DIGITAL_CORE / well_name
    foto_dir = next((d for d in well_dir.iterdir() if d.is_dir() and d.name.lower() == 'фото'), None)
    if foto_dir is None:
        return []
    ds_dir = foto_dir / 'ДС'
    uv_dir = foto_dir / 'УФ'
    if not ds_dir.exists():
        return []
    found = []
    for ds_path in sorted(ds_dir.glob('*.jpeg')):
        d = _parse_depth(ds_path.stem)   # stem = без расширения
        if d is None:
            continue
        d_from, d_to = d
        if target_depth is not None and abs(d_from - target_depth) > 0.05:
            continue
        uv_path = None
        if uv_dir.exists():
            uv_path = next(
                (p for p in uv_dir.glob('*.jpeg')
                 if abs((_parse_depth(p.stem) or (0, 0))[0] - d_from) < 0.05),
                None,
            )
        found.append((ds_path, uv_path, d_from, d_to))
    return found


def _load_markup(well_name: str) -> pd.DataFrame:
    csv_path = _cfg.PIPELINE_CSV / f'{well_name}.csv'
    df = pd.read_csv(csv_path)
    df['depth_from'] = pd.to_numeric(df['depth_from'], errors='coerce')
    df['depth_to']   = pd.to_numeric(df['depth_to'],   errors='coerce')
    return df.dropna(subset=['depth_from', 'depth_to']).sort_values('depth_from').reset_index(drop=True)


@torch.no_grad()
def _predict_single(img: np.ndarray, model: torch.nn.Module, tfm, depth_from: float, depth_to: float):
    h_px    = img.shape[0]
    span_cm = (depth_to - depth_from) * 100
    tile_px = max(1, int(round(h_px / span_cm * _cfg.TILE_CM)))
    preds, y, d = [], 0, depth_from
    while y + tile_px <= h_px:
        tensor = tfm(Image.fromarray(img[y: y + tile_px])).unsqueeze(0).to(DEVICE)
        probs  = F.softmax(model(tensor), dim=1).squeeze(0).cpu().numpy()
        idx    = int(probs.argmax())
        preds.append({'depth_from': d, 'depth_to': d + _cfg.TILE_CM / 100,
                      'class_idx': idx, 'conf': float(probs[idx])})
        y += tile_px
        d += _cfg.TILE_CM / 100
    return preds


@torch.no_grad()
def _predict_dual(img_ds: np.ndarray, img_uv: np.ndarray,
                  model: DualStreamResNet18, tfm_ds, tfm_uv,
                  depth_from: float, depth_to: float):
    h_px    = img_ds.shape[0]
    span_cm = (depth_to - depth_from) * 100
    tile_px = max(1, int(round(h_px / span_cm * _cfg.TILE_CM)))
    preds, y, d = [], 0, depth_from
    while y + tile_px <= h_px:
        t_ds = tfm_ds(Image.fromarray(img_ds[y: y + tile_px])).unsqueeze(0).to(DEVICE)
        t_uv = tfm_uv(Image.fromarray(img_uv[y: y + tile_px])).unsqueeze(0).to(DEVICE)
        probs = F.softmax(model(t_ds, t_uv), dim=1).squeeze(0).cpu().numpy()
        idx   = int(probs.argmax())
        preds.append({'depth_from': d, 'depth_to': d + _cfg.TILE_CM / 100,
                      'class_idx': idx, 'conf': float(probs[idx])})
        y += tile_px
        d += _cfg.TILE_CM / 100
    return preds


def _draw_strip(ax, segments, z_top, z_bot, title='', show_conf=False):
    ax.set_xlim(0, 1); ax.set_ylim(z_bot, z_top)
    for y0, y1, color, conf in segments:
        ax.fill_betweenx([y0, y1], 0.05, 0.95, color=color, linewidth=0)
        if show_conf and conf is not None and conf > 0.65 and (y1 - y0) > 0.03:
            ax.text(0.5, (y0 + y1) / 2, f'{conf:.0%}', ha='center', va='center',
                    fontsize=6, color='white', fontweight='bold')
    ax.set_xticks([]); ax.set_title(title, fontsize=10, pad=4)
    for sp in ax.spines.values():
        sp.set_visible(False)


# ─────────────────────────────────────────────────────────────────────────────
# Визуализация одной фотографии
# ─────────────────────────────────────────────────────────────────────────────

def visualize_photo(well_name: str, models_dict: dict, tfms_dict: dict,
                    idx2name: dict, idx2color: dict, vis_dir: Path,
                    target_depth=None, dual_model=None, tfm_ds_dual=None, tfm_uv_dual=None):
    photos    = _find_photos(well_name, target_depth)
    markup_df = _load_markup(well_name)
    if not photos:
        print(f'  [SKIP] {well_name}: фото не найдено (depth={target_depth})')
        return

    for ds_path, uv_path, d_from, d_to in photos:
        img_ds = np.array(Image.open(ds_path).convert('RGB'))
        img_uv = np.array(Image.open(uv_path).convert('RGB')) if uv_path else img_ds

        # Предсказания
        if dual_model is not None:
            preds_dual = _predict_dual(img_ds, img_uv, dual_model, tfm_ds_dual, tfm_uv_dual, d_from, d_to)
            strips = {'Dual ДС+УФ': preds_dual}
        else:
            strips = {}
            for mod, model in models_dict.items():
                img = img_ds if mod == 'ДС' else img_uv
                strips[f'Модель {mod}'] = _predict_single(img, model, tfms_dict[mod], d_from, d_to)

        # Разметка геолога
        sub = markup_df[(markup_df.depth_from < d_to) & (markup_df.depth_to > d_from)]
        markup_segs = []
        for _, row in sub.iterrows():
            y0  = max(row['depth_from'], d_from)
            y1  = min(row['depth_to'],   d_to)
            nm  = str(row['mineral']).strip().replace(' ', '_')
            col = _cfg.CLASS_PALETTE.get(nm, _cfg.CLASS_PALETTE.get(str(row['mineral']).strip(), _cfg.CLASS_PALETTE['unknown']))
            if y1 - y0 > 0.001:
                markup_segs.append((y0, y1, col, None))

        n_strips  = len(strips)
        width_ratios = [2.5, 2.5] + [0.75] * n_strips + [0.75, 2.8]
        fig = plt.figure(figsize=(5 * (3 + n_strips), 11))
        gs  = plt.GridSpec(1, 3 + n_strips + 1, figure=fig,
                           width_ratios=width_ratios, wspace=0.10)

        extent = [0, 1, d_to, d_from]
        ax_ds = fig.add_subplot(gs[0, 0])
        ax_ds.imshow(img_ds, aspect='auto', extent=extent)
        ax_ds.set_title('Фото ДС', fontsize=11); ax_ds.set_xticks([])
        ax_ds.yaxis.set_tick_params(labelsize=8)

        ax_uv = fig.add_subplot(gs[0, 1])
        ax_uv.imshow(img_uv, aspect='auto', extent=extent)
        ax_uv.set_title('Фото УФ', fontsize=11); ax_uv.set_xticks([]); ax_uv.set_yticks([])

        shown_cls: Dict[str, str] = {}
        for si, (strip_title, preds) in enumerate(strips.items()):
            ax = fig.add_subplot(gs[0, 2 + si])
            segs = [(p['depth_from'], p['depth_to'], idx2color[p['class_idx']], p['conf'])
                    for p in preds]
            _draw_strip(ax, segs, d_from, d_to, strip_title, show_conf=True)
            ax.set_yticks([])
            for p in preds:
                shown_cls[idx2name[p['class_idx']]] = idx2color[p['class_idx']]

        ax_markup = fig.add_subplot(gs[0, 2 + n_strips])
        _draw_strip(ax_markup, markup_segs, d_from, d_to, 'Разметка\nгеолога')
        ax_markup.set_yticks([])
        for y0, y1, col, _ in markup_segs:
            shown_cls[str(col)] = col

        ax_leg = fig.add_subplot(gs[0, 3 + n_strips])
        ax_leg.axis('off'); ax_leg.set_title('Легенда', fontsize=11, fontweight='bold')
        handles = [mpatches.Patch(facecolor=color, edgecolor='#555', label=name)
                   for name, color in sorted(shown_cls.items()) if not name.startswith('#')]
        ax_leg.legend(handles=handles, loc='upper left', fontsize=9, frameon=True,
                      bbox_to_anchor=(0.0, 1.0))

        fig.suptitle(f'{well_name}  |  {d_from:.2f} – {d_to:.2f} м',
                     fontsize=13, fontweight='bold', y=1.01)
        plt.tight_layout()
        safe = well_name.replace(' ', '_')
        out  = vis_dir / f'{safe}_{d_from:.2f}-{d_to:.2f}_full.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Сохранено: {out}')


# ─────────────────────────────────────────────────────────────────────────────
# Статистика на сплите
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_predictions_single(model, modality: str, split: str):
    cfg = _cfg.MODEL_CONFIGS[modality]
    mean, std = (_cfg.DS_MEAN, _cfg.DS_STD) if modality == 'ДС' else (_cfg.UV_MEAN, _cfg.UV_STD)
    ds = tv_datasets.ImageFolder(
        _cfg.DATASET_ROOT / modality / split,
        transform=get_transforms(cfg['resize'], 'none', is_train=False, mean=mean, std=std),
    )
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4,
                        pin_memory=(DEVICE.type == 'cuda'))
    all_labels, all_preds, all_probs = [], [], []
    for inputs, labels in loader:
        probs = F.softmax(model(inputs.to(DEVICE)), dim=1).cpu().numpy()
        all_labels.extend(labels.numpy())
        all_preds.extend(probs.argmax(axis=1))
        all_probs.append(probs)
    return np.array(all_labels), np.array(all_preds), np.vstack(all_probs), ds.classes


@torch.no_grad()
def collect_predictions_dual(model, split: str):
    from src.data import PairedDataset
    cfg_ds = _cfg.MODEL_CONFIGS['ДС']
    cfg_uv = _cfg.MODEL_CONFIGS['УФ']
    ds = PairedDataset(
        _cfg.DATASET_ROOT / 'ДС' / split,
        _cfg.DATASET_ROOT / 'УФ' / split,
        get_transforms(cfg_ds['resize'], 'none', False, _cfg.DS_MEAN, _cfg.DS_STD),
        get_transforms(cfg_uv['resize'], 'none', False, _cfg.UV_MEAN, _cfg.UV_STD),
    )
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4,
                        pin_memory=(DEVICE.type == 'cuda'))
    all_labels, all_preds, all_probs = [], [], []
    for (x_ds, x_uv), labels in loader:
        probs = F.softmax(model(x_ds.to(DEVICE), x_uv.to(DEVICE)), dim=1).cpu().numpy()
        all_labels.extend(labels.numpy())
        all_preds.extend(probs.argmax(axis=1))
        all_probs.append(probs)
    return np.array(all_labels), np.array(all_preds), np.vstack(all_probs), ds.classes


def plot_stats(results: dict, stats_dir: Path) -> None:
    stats_dir.mkdir(parents=True, exist_ok=True)

    # Confusion matrices
    n_mods = len(results)
    fig, axes = plt.subplots(1, n_mods, figsize=(9 * n_mods, 7))
    if n_mods == 1:
        axes = [axes]
    for ax, (mod_name, res) in zip(axes, results.items()):
        short = [_cfg.CLASS_SHORT.get(c, c[:8]) for c in res['classes']]
        cm    = confusion_matrix(res['labels'], res['preds'],
                                 labels=list(range(len(res['classes']))), normalize='true')
        ConfusionMatrixDisplay(cm, display_labels=short).plot(
            ax=ax, colorbar=True, xticks_rotation=40, values_format='.2f')
        ax.set_title(f'Confusion matrix — {mod_name}', fontsize=11)
    plt.suptitle('Confusion Matrix', fontsize=13)
    plt.tight_layout()
    fig.savefig(stats_dir / 'confusion_test.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  {stats_dir}/confusion_test.png')

    # Per-class F1
    fig, axes = plt.subplots(1, n_mods, figsize=(9 * n_mods, 5), sharey=True)
    if n_mods == 1:
        axes = [axes]
    for ax, (mod_name, res) in zip(axes, results.items()):
        class_names = res['classes']
        short  = [_cfg.CLASS_SHORT.get(c, c[:10]) for c in class_names]
        colors = [_cfg.CLASS_PALETTE.get(c, '#94A3B8') for c in class_names]
        f1_per = f1_score(res['labels'], res['preds'],
                          labels=list(range(len(class_names))), average=None, zero_division=0)
        macro  = f1_per.mean()
        bars = ax.bar(short, f1_per, color=colors, edgecolor='#333', linewidth=0.6)
        for bar, val in zip(bars, f1_per):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.axhline(macro, color='#64748B', linestyle='--', linewidth=1.5,
                   label=f'macro={macro:.3f}')
        ax.set_title(f'F1 по классам — {mod_name}', fontsize=11)
        ax.set_ylim(0, 1.08); ax.tick_params(axis='x', rotation=35); ax.legend()
        for s in ['top', 'right']:
            ax.spines[s].set_visible(False)
    plt.tight_layout()
    fig.savefig(stats_dir / 'per_class_f1_test.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  {stats_dir}/per_class_f1_test.png')

    # Confidence distribution
    fig, axes = plt.subplots(1, n_mods, figsize=(8 * n_mods, 5))
    if n_mods == 1:
        axes = [axes]
    for ax, (mod_name, res) in zip(axes, results.items()):
        max_p   = res['probs'].max(axis=1)
        correct = (res['labels'] == res['preds'])
        ax.hist(max_p[correct],  bins=25, alpha=0.75, color='#22C55E',
                label=f'Верно ({correct.sum():,})', density=True)
        ax.hist(max_p[~correct], bins=25, alpha=0.75, color='#EF4444',
                label=f'Ошибка ({(~correct).sum():,})', density=True)
        ax.axvline(max_p.mean(), color='#64748B', linestyle='--',
                   label=f'Среднее={max_p.mean():.2f}')
        ax.set_title(f'Уверенность — {mod_name}'); ax.set_xlim(0, 1); ax.legend()
    plt.tight_layout()
    fig.savefig(stats_dir / 'confidence_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  {stats_dir}/confidence_distribution.png')

    # Classification report table
    fig, axes = plt.subplots(1, n_mods, figsize=(11 * n_mods, 5))
    if n_mods == 1:
        axes = [axes]
    for ax, (mod_name, res) in zip(axes, results.items()):
        class_names = res['classes']
        short = [_cfg.CLASS_SHORT.get(c, c[:10]) for c in class_names]
        prec, rec, f1_per, support = precision_recall_fscore_support(
            res['labels'], res['preds'], labels=list(range(len(class_names))), zero_division=0)
        macro_f1 = f1_score(res['labels'], res['preds'], average='macro', zero_division=0)
        acc      = accuracy_score(res['labels'], res['preds'])
        rows = [[sn, f'{p:.2f}', f'{r:.2f}', f'{f:.2f}', f'{s:,}']
                for sn, p, r, f, s in zip(short, prec, rec, f1_per, support)]
        rows.append(['macro avg', '', '', f'{macro_f1:.2f}', ''])
        rows.append([f'Accuracy {acc:.3f}', '', '', '', f'{len(res["labels"]):,}'])

        ax.axis('off')
        tbl = ax.table(cellText=rows, colLabels=['Класс', 'Prec', 'Rec', 'F1', 'Support'],
                       cellLoc='center', loc='center')
        tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.0, 1.5)
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_facecolor('#1E3A5F'); cell.set_text_props(color='white', fontweight='bold')
            elif r >= len(class_names) + 1:
                cell.set_facecolor('#E2E8F0'); cell.set_text_props(fontweight='bold')
            elif r % 2 == 0:
                cell.set_facecolor('#F8FAFC')
            cell.set_edgecolor('#CBD5E1')
        ax.set_title(f'Classification Report — {mod_name}', fontsize=11, pad=12)

        print(f'\n{mod_name}:')
        print(classification_report(res['labels'], res['preds'],
                                    target_names=class_names, zero_division=0))

    plt.tight_layout()
    fig.savefig(stats_dir / 'classification_report.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  {stats_dir}/classification_report.png')


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Визуализация результатов ML-Core-Kern-',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('run_name', help='Имя папки в results/ (например: run_v4)')
    p.add_argument('--dual',      action='store_true',
                   help='Использовать dual_best.pth вместо ДС/УФ отдельных моделей')
    p.add_argument('--split',     default='test', choices=['train', 'val', 'test'],
                   help='Сплит для метрик (default: test)')
    p.add_argument('--well',      default=None,
                   help='Имя скважины для визуализации (default: тест-скважины из manifest)')
    p.add_argument('--depth',     type=float, default=None,
                   help='depth_from фото для --well (default: все фото скважины)')
    p.add_argument('--wells',     type=int, default=0,
                   help='Кол-во скважин для per-well визуализации (0 = все тест-скважины, -1 = пропустить)')
    p.add_argument('--no-stats',  action='store_true', help='Пропустить confusion matrix и F1 графики')
    p.add_argument('--no-photos', action='store_true', help='Пропустить per-well визуализацию')
    return p.parse_args()


def main():
    args = parse_args()
    run_dir   = _cfg.RESULTS_ROOT / args.run_name
    vis_dir   = run_dir / 'visual'
    stats_dir = vis_dir / 'stats'
    vis_dir.mkdir(parents=True, exist_ok=True)

    arch = _read_arch(run_dir)

    print(f'Run    : {run_dir}')
    print(f'Arch   : {arch}')
    print(f'Split  : {args.split}')
    print(f'Dual   : {args.dual}')
    print(f'Device : {DEVICE}')

    # ── Загрузка моделей ──────────────────────────────────────────────────────
    models_dict: dict = {}
    tfms_dict:   dict = {}
    dual_model        = None
    tfm_ds_dual = tfm_uv_dual = None

    if args.dual:
        pth = run_dir / 'dual_best.pth'
        if not pth.exists():
            sys.exit(f'Не найден: {pth}')
        dual_model = _load_dual(pth, arch=arch)
        cfg_ds = _cfg.MODEL_CONFIGS['ДС']
        cfg_uv = _cfg.MODEL_CONFIGS['УФ']
        tfm_ds_dual = get_transforms(cfg_ds['resize'], 'none', False, _cfg.DS_MEAN, _cfg.DS_STD)
        tfm_uv_dual = get_transforms(cfg_uv['resize'], 'none', False, _cfg.UV_MEAN, _cfg.UV_STD)
        print(f'Загружена dual-модель: {pth.name}')
    else:
        for mod in _cfg.MODALITIES:
            pth = run_dir / f'{mod}_best.pth'
            if not pth.exists():
                print(f'  [SKIP] {pth} не найден')
                continue
            models_dict[mod] = _load_single(pth, mod, arch=arch)
            cfg = _cfg.MODEL_CONFIGS[mod]
            mean, std = (_cfg.DS_MEAN, _cfg.DS_STD) if mod == 'ДС' else (_cfg.UV_MEAN, _cfg.UV_STD)
            tfms_dict[mod] = get_transforms(cfg['resize'], 'none', False, mean, std)
            print(f'Загружена модель {mod}: {pth.name}')

    if not models_dict and dual_model is None:
        sys.exit('Нет загруженных моделей — выход.')

    # IDX2NAME из первого доступного checkpoint
    if args.dual:
        state = torch.load(run_dir / 'dual_best.pth', map_location='cpu', weights_only=True)
        n_cls = _num_classes_from_state(state, dual=True)
    else:  # noqa: E501
        first_mod = next(iter(models_dict))
        pth = run_dir / f'{first_mod}_best.pth'
        state = torch.load(pth, map_location='cpu', weights_only=True)
        n_cls = _num_classes_from_state(state, dual=False)

    # Определяем классы: из label_encoder.json (или fallback на CLASSES_ORDER[:n_cls])
    le_path = _cfg.LABEL_ENCODER
    if le_path.exists():
        with open(le_path, encoding='utf-8') as f:
            raw_enc = json.load(f)
        idx2name = {v: k for k, v in raw_enc.items()}
    else:
        idx2name = {i: _cfg.CLASSES_ORDER[i] for i in range(min(n_cls, len(_cfg.CLASSES_ORDER)))}

    idx2color = {i: _cfg.CLASS_PALETTE.get(name, _cfg.CLASS_PALETTE['unknown'])
                 for i, name in idx2name.items()}
    print(f'Классов: {n_cls}  →  {[idx2name.get(i,"?") for i in range(n_cls)]}')

    # ── Per-well визуализация ─────────────────────────────────────────────────
    if not args.no_photos:
        if args.well:
            wells_to_show = [args.well]
        else:
            manifest_path = _cfg.DATASET_ROOT / 'split_manifest.json'
            if manifest_path.exists():
                with open(manifest_path, encoding='utf-8') as f:
                    manifest = json.load(f)
                wells_to_show = [w for w, v in manifest.items() if v.get('split') == args.split]
            else:
                wells_to_show = []

        if args.wells > 0:
            wells_to_show = wells_to_show[:args.wells]
        elif args.wells == -1:
            wells_to_show = []

        if wells_to_show:
            print(f'\nВизуализация скважин: {wells_to_show}')
            for well in wells_to_show:
                print(f'  {well} ...')
                visualize_photo(
                    well, models_dict, tfms_dict, idx2name, idx2color, vis_dir,
                    target_depth=args.depth,
                    dual_model=dual_model, tfm_ds_dual=tfm_ds_dual, tfm_uv_dual=tfm_uv_dual,
                )

    # ── Метрики ───────────────────────────────────────────────────────────────
    if not args.no_stats:
        print(f'\nСбор предсказаний на {args.split}-сете...')
        results = {}

        if args.dual:
            labels, preds, probs, classes = collect_predictions_dual(dual_model, args.split)
            f1  = f1_score(labels, preds, average='macro', zero_division=0)
            acc = accuracy_score(labels, preds)
            print(f'  Dual: F1={f1:.4f}  Acc={acc:.4f}  (n={len(labels)})')
            results['Dual ДС+УФ'] = {'labels': labels, 'preds': preds,
                                      'probs': probs, 'classes': classes}
        else:
            for mod, model in models_dict.items():
                labels, preds, probs, classes = collect_predictions_single(model, mod, args.split)
                f1  = f1_score(labels, preds, average='macro', zero_division=0)
                acc = accuracy_score(labels, preds)
                print(f'  {mod}: F1={f1:.4f}  Acc={acc:.4f}  (n={len(labels)})')
                results[mod] = {'labels': labels, 'preds': preds,
                                'probs': probs, 'classes': classes}

        print(f'\nСохраняем графики в {stats_dir}:')
        plot_stats(results, stats_dir)

    print('\nГотово.')


if __name__ == '__main__':
    main()
