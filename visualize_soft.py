"""
Визуализация результатов run_soft_dual (мягкие метки, 5 базовых компонент).

Каждый столб керна показывает 6 панелей:
  ДС фото | УФ фото | Геолог (CSV) | 6-классов | Soft-истина | Soft-предсказание + энтропия

Статистика:
  1. Матрица ошибок 5×5      4. Отчёт по классам
  2. F1 по компонентам       5. Энтропия по истинному классу
  3. Средний состав          6. KL-дивергенция предсказания

Запуск:
    python3 visualize_soft.py
    python3 visualize_soft.py --run run_soft_dual --split test
    python3 visualize_soft.py --no-columns   # только stats
    python3 visualize_soft.py --well Харасавэйск_700 --depth 800
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    f1_score, precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
import config as _cfg
from src.transforms import get_transforms
from src.data import SoftLabelPairedDataset
from src.models.dual_resnet import DualStreamResNet18

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Цвета базовых компонент
COMP_COLORS = {
    'Песчаник':          '#3B82F6',   # синий
    'Алевролит':         '#F59E0B',   # янтарный
    'Аргиллит':          '#EF4444',   # красный
    'Карбонат':          '#10B981',   # изумрудный
    'Глина_опоковидная': '#8B5CF6',   # фиолетовый
}
COMP_SHORT = {
    'Песчаник': 'Пес', 'Алевролит': 'Алев',
    'Аргиллит': 'Арг', 'Карбонат': 'Карб',
    'Глина_опоковидная': 'Гл.оп',
}

FRAG_RE = re.compile(r'_(\d+\.\d{3})_(\d+\.\d{3})_frag\d{4}\.jpg$')
_DEPTH_RE = re.compile(r'_([\d]+[.,][\d]+)\s*-\s*([\d]+[.,][\d]+)$')

# ── Цвета 6-классового маппинга ───────────────────────────────────────────────
SIX_CLASS_COLORS = {
    'Песчаник':      '#3B82F6',
    'Аргиллит':      '#EF4444',
    'Алевролит':     '#F59E0B',
    'Перес_светлое': '#8B5CF6',
    'Перес_тёмное':  '#4C1D95',
    'Карбонат':      '#10B981',
    'unknown':       '#94A3B8',
}

# ── 6-классовый маппинг (из prepare_exp.py, ключи с пробелами из CSV) ────────
# Нормализация: CSV-имена с пробелами → ключи с подчёркиваниями
def _csv_mineral_key(name: str) -> str:
    """CSV: 'Глина аргиллитоподобная' → 'Глина_аргиллитоподобная'"""
    return name.strip().replace(' ', '_')

MINERAL_TO_6CLASS = {
    'Аргиллит': 'Аргиллит', 'Аргиллит_углистый': 'Аргиллит',
    'Аргиллит_с_включениями_угля': 'Аргиллит', 'Уголь': 'Аргиллит',
    'Уголь_с_прослоями_аргиллита': 'Аргиллит',
    'Алевролит': 'Алевролит', 'Алевролит_с_включениями_угля': 'Алевролит',
    'Алевролит_глинистый': 'Алевролит', 'Алевролит_карбонатный': 'Алевролит',
    'Глинисто-карбонатная_порода': 'Карбонат', 'Опока_глинистая': 'Карбонат',
    'Глина_опоковидная': 'Карбонат',
    'Глина_аргиллитоподобная_с_прослоями_глины_опоковидной': 'Карбонат',
    'Кремнисто-глинистая_порода': 'Карбонат',
    'Глина_опоковидная_с_включением_глинистых_опок': 'Карбонат',
    'Глина_аргиллитоподобная': 'Карбонат',
    'Переслаивание_песчаника,_аргиллита_и_алевролита': 'Перес_светлое',
    'Песчаник_с_включениями_алевролита_и_аргиллита': 'Перес_светлое',
    'Чередование_аргиллита,_алевролита_и_песчаника': 'Перес_светлое',
    'Алевролит_с_прослоями_песчаника_и_аргиллита': 'Перес_светлое',
    'Песчаник_с_прослоями_алевролита_и_аргиллита': 'Перес_светлое',
    'Аргиллит_с_прослоями_песчаника_и_алевролита': 'Перес_светлое',
    'Песчаник_с_прослоями_алевролита': 'Перес_светлое',
    'Переслаивание_песчаника_и_алевролита': 'Перес_светлое',
    'Алевролит_с_прослоями_песчаника': 'Перес_светлое',
    'Песчаник_с_прослоями_аргиллита': 'Перес_светлое',
    'Аргиллит_с_прослоями_песчаника': 'Перес_светлое',
    'Переслаивание_песчаника_и_аргиллита': 'Перес_светлое',
    'Переслаивание_аргиллита_и_алевролита': 'Перес_тёмное',
    'Аргиллит_алевритовый': 'Перес_тёмное',
    'Алевролит_с_прослоями_аргиллита': 'Перес_тёмное',
    'Аргиллит_с_прослоями_алевролита': 'Перес_тёмное',
    'Песчаник': 'Песчаник', 'Песчаник_карбонатный': 'Песчаник',
}

# Уникальные цвета для оригинальных минералов в CSV (автоматически)
import colorsys as _cs
def _mineral_color(mineral_key: str) -> str:
    """Детерминированный цвет по хешу имени минерала."""
    h = abs(hash(mineral_key)) % 360 / 360.0
    r, g, b = _cs.hls_to_rgb(h, 0.55, 0.65)
    return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'


# ─────────────────────────────────────────────────────────────────────────────
# Загрузка модели и данных
# ─────────────────────────────────────────────────────────────────────────────

def load_model(results_dir: Path, num_classes: int = 5) -> torch.nn.Module:
    """Загружает модель нужной архитектуры из config.json."""
    cfg_path = results_dir / 'config.json'
    arch = 'resnet18'
    if cfg_path.exists():
        with open(cfg_path) as f:
            arch = json.load(f).get('arch', 'resnet18').replace('dual_', '')

    ckpt = torch.load(results_dir / 'dual_best.pth', map_location=DEVICE)

    if arch in ('resnet50',):
        from src.models.dual_resnet import create_dual_resnet50
        model = create_dual_resnet50(num_classes, dropout_p=0.5)
    elif arch in ('efficientnet_b3', 'efficientnet_b4', 'convnext_tiny', 'swin_t'):
        from src.models.backbones import create_dual
        model = create_dual(arch, num_classes, dropout_p=0.5)
    else:
        model = DualStreamResNet18(num_classes=num_classes)

    model.load_state_dict(ckpt)
    return model.to(DEVICE).eval()


@torch.no_grad()
def collect_test_predictions(model, dataset_root: Path, split: str,
                              soft_labels: dict, base_classes: list,
                              ds_mean, ds_std, uv_mean, uv_std):
    ds_tfm = get_transforms('square', 'none', False, ds_mean, ds_std)
    uv_tfm = get_transforms('square', 'none', False, uv_mean, uv_std)
    ds = SoftLabelPairedDataset(
        dataset_root / 'ДС' / split,
        dataset_root / 'УФ' / split,
        ds_tfm, uv_tfm, soft_labels,
    )
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4,
                        pin_memory=(DEVICE.type == 'cuda'))
    all_probs, all_true_soft, all_true_hard = [], [], []
    for (x_ds, x_uv), soft_lbl in tqdm(loader, desc=f'Inference {split}'):
        probs = F.softmax(model(x_ds.to(DEVICE), x_uv.to(DEVICE)), dim=1).cpu().numpy()
        all_probs.append(probs)
        all_true_soft.append(soft_lbl.numpy())
        all_true_hard.extend(soft_lbl.argmax(dim=1).numpy())
    probs_arr      = np.vstack(all_probs)
    true_soft_arr  = np.vstack(all_true_soft)
    true_hard_arr  = np.array(all_true_hard)
    pred_hard_arr  = probs_arr.argmax(axis=1)
    return probs_arr, true_soft_arr, true_hard_arr, pred_hard_arr, ds.classes


# ─────────────────────────────────────────────────────────────────────────────
# Статистические графики
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion(true_hard, pred_hard, base_classes, out_path: Path):
    short = [COMP_SHORT.get(c, c[:6]) for c in base_classes]
    cm = confusion_matrix(true_hard, pred_hard, labels=list(range(len(base_classes))),
                          normalize='true')
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=short)
    disp.plot(ax=ax, colorbar=True, xticks_rotation=30, values_format='.0%')
    ax.set_title('Матрица ошибок — базовые компоненты\n(нормировано по строкам)', fontsize=12)
    ax.set_xlabel('Предсказано'); ax.set_ylabel('Истинный класс')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  {out_path.name}')


def plot_f1_bars(true_hard, pred_hard, base_classes, out_path: Path):
    f1_per = f1_score(true_hard, pred_hard, labels=list(range(len(base_classes))),
                      average=None, zero_division=0)
    prec, rec, _, support = precision_recall_fscore_support(
        true_hard, pred_hard, labels=list(range(len(base_classes))), zero_division=0)
    short  = [COMP_SHORT.get(c, c[:8]) for c in base_classes]
    colors = [COMP_COLORS.get(c, '#94A3B8') for c in base_classes]
    macro  = f1_per.mean()

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(short, f1_per, color=colors, edgecolor='#333', linewidth=0.7, width=0.6)
    for bar, val, sup in zip(bars, f1_per, support):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.text(bar.get_x() + bar.get_width() / 2, 0.01,
                f'n={sup:,}', ha='center', va='bottom', fontsize=8, color='#555')
    ax.axhline(macro, color='#64748B', linestyle='--', linewidth=1.5,
               label=f'macro F1 = {macro:.3f}')
    ax.set_title('F1 по базовым компонентам — тест', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.15); ax.legend(fontsize=10)
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  {out_path.name}')


def plot_avg_composition(probs_arr, true_hard, base_classes, out_path: Path):
    """Для каждого истинного класса — средний предсказанный вектор состава."""
    n = len(base_classes)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(n)
    width = 0.7

    avg_comps = []
    supports  = []
    for i in range(n):
        mask = true_hard == i
        avg = probs_arr[mask].mean(axis=0) if mask.sum() > 0 else np.zeros(n)
        avg_comps.append(avg)
        supports.append(mask.sum())

    bottoms = np.zeros(n)
    for j, comp in enumerate(base_classes):
        vals   = [avg_comps[i][j] for i in range(n)]
        color  = COMP_COLORS.get(comp, '#94A3B8')
        ax.bar(x, vals, width, bottom=bottoms, color=color, label=COMP_SHORT.get(comp, comp),
               edgecolor='white', linewidth=0.5)
        # Подписи для долей > 5%
        for xi, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 0.05:
                ax.text(xi, b + v / 2, f'{v:.0%}', ha='center', va='center',
                        fontsize=8, color='white', fontweight='bold')
        bottoms = bottoms + np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f'{COMP_SHORT.get(c,c)}\n(n={supports[i]:,})' for i, c in enumerate(base_classes)],
        fontsize=9)
    ax.set_ylabel('Доля компоненты в предсказании')
    ax.set_title('Средний предсказанный состав по истинному классу\n'
                 '(показывает насколько модель понимает "смешанность" класса)',
                 fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  {out_path.name}')


def plot_confidence(probs_arr, true_hard, pred_hard, out_path: Path):
    max_p   = probs_arr.max(axis=1)
    correct = (true_hard == pred_hard)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(max_p[correct],  bins=30, alpha=0.75, color='#22C55E',
            label=f'Верно ({correct.sum():,})', density=True)
    ax.hist(max_p[~correct], bins=30, alpha=0.75, color='#EF4444',
            label=f'Ошибка ({(~correct).sum():,})', density=True)
    ax.axvline(max_p.mean(), color='#64748B', linestyle='--',
               label=f'Среднее = {max_p.mean():.2f}')
    ax.set_title('Уверенность модели (max softmax probability)', fontsize=12)
    ax.set_xlabel('Уверенность'); ax.set_xlim(0, 1); ax.legend()
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  {out_path.name}')


def plot_entropy_by_class(probs_arr, true_hard, base_classes, out_path: Path):
    """Распределение энтропии предсказания по истинному классу."""
    n = len(base_classes)
    p_clipped = np.clip(probs_arr, 1e-9, 1)
    entropies  = -np.sum(p_clipped * np.log2(p_clipped), axis=1) / np.log2(n)  # 0..1

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Boxplot энтропии по классу
    ax = axes[0]
    data   = [entropies[true_hard == i] for i in range(n)]
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops={'color': 'white', 'linewidth': 2})
    for patch, cls in zip(bp['boxes'], base_classes):
        patch.set_facecolor(COMP_COLORS.get(cls, '#94A3B8'))
        patch.set_alpha(0.85)
    ax.set_xticklabels([COMP_SHORT.get(c, c) for c in base_classes], fontsize=9)
    ax.set_ylabel('Нормализованная энтропия H (0=уверен, 1=равновероятно)')
    ax.set_title('Неуверенность модели по истинному классу\n'
                 '(высокая H = модель не знает что это)', fontsize=10, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5,
               label='H=0.5 (умеренная)')
    ax.legend(fontsize=8)
    for s in ['top', 'right']: ax.spines[s].set_visible(False)

    # Гистограмма энтропии: правильные vs ошибочные
    ax = axes[1]
    correct = (true_hard == probs_arr.argmax(axis=1))
    ax.hist(entropies[correct],  bins=25, alpha=0.75, color='#22C55E',
            density=True, label=f'Верно ({correct.sum():,})')
    ax.hist(entropies[~correct], bins=25, alpha=0.75, color='#EF4444',
            density=True, label=f'Ошибка ({(~correct).sum():,})')
    ax.axvline(entropies.mean(), color='#64748B', linestyle='--',
               label=f'Среднее H={entropies.mean():.2f}')
    ax.set_xlabel('Энтропия H'); ax.set_ylabel('Плотность')
    ax.set_title('Энтропия: верные vs ошибочные предсказания\n'
                 '(ошибки концентрируются при высокой H)', fontsize=10, fontweight='bold')
    ax.set_xlim(0, 1); ax.legend(fontsize=9)
    for s in ['top', 'right']: ax.spines[s].set_visible(False)

    plt.suptitle('Неуверенность (энтропия) предсказаний — Soft Labels',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  {out_path.name}')


def plot_kl_divergence(probs_arr, true_soft_arr, true_hard, base_classes, out_path: Path):
    """KL-дивергенция между предсказанным и истинным soft-распределением."""
    p_clipped = np.clip(probs_arr,     1e-9, 1)
    q_clipped = np.clip(true_soft_arr, 1e-9, 1)
    # KL(true || pred) = sum(q * log(q/p))
    kl_vals = np.sum(q_clipped * np.log2(q_clipped / p_clipped), axis=1)
    kl_vals = np.clip(kl_vals, 0, 10)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # KL по истинному классу
    ax = axes[0]
    data = [kl_vals[true_hard == i] for i in range(len(base_classes))]
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops={'color': 'white', 'linewidth': 2})
    for patch, cls in zip(bp['boxes'], base_classes):
        patch.set_facecolor(COMP_COLORS.get(cls, '#94A3B8'))
        patch.set_alpha(0.85)
    ax.set_xticklabels([COMP_SHORT.get(c, c) for c in base_classes], fontsize=9)
    ax.set_ylabel('KL(истина || предсказание), bits')
    ax.set_title('KL-дивергенция по классу\n'
                 '(0 = идеал, чем меньше — тем точнее состав)', fontsize=10, fontweight='bold')
    for s in ['top', 'right']: ax.spines[s].set_visible(False)

    # Scatter: KL vs уверенность (max prob)
    ax = axes[1]
    max_p = probs_arr.max(axis=1)
    correct = (true_hard == probs_arr.argmax(axis=1))
    colors  = np.where(correct, '#22C55E', '#EF4444')
    ax.scatter(max_p, kl_vals, c=colors, alpha=0.3, s=8, linewidths=0)
    ax.set_xlabel('Уверенность модели (max softmax)')
    ax.set_ylabel('KL-дивергенция (bits)')
    ax.set_title('Уверенность vs точность состава\n'
                 '(зел. = argmax верный, крас. = argmax ошибочный)',
                 fontsize=10, fontweight='bold')
    # Добавляем аннотацию
    ax.text(0.95, 0.95, f'Median KL={np.median(kl_vals):.2f}\nMean KL={kl_vals.mean():.2f}',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='#F1F5F9', alpha=0.8))
    for s in ['top', 'right']: ax.spines[s].set_visible(False)

    plt.suptitle('KL-дивергенция: насколько точно модель угадывает состав',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  {out_path.name}')


def plot_report_table(true_hard, pred_hard, base_classes, out_path: Path):
    prec, rec, f1_per, sup = precision_recall_fscore_support(
        true_hard, pred_hard, labels=list(range(len(base_classes))), zero_division=0)
    macro_f1  = f1_score(true_hard, pred_hard, average='macro', zero_division=0)
    weight_f1 = f1_score(true_hard, pred_hard, average='weighted', zero_division=0)

    rows = []
    for i, cls in enumerate(base_classes):
        rows.append([cls, f'{prec[i]:.3f}', f'{rec[i]:.3f}', f'{f1_per[i]:.3f}', f'{int(sup[i]):,}'])
    rows.append(['macro avg',    '—', '—', f'{macro_f1:.3f}',  f'{int(sum(sup)):,}'])
    rows.append(['weighted avg', '—', '—', f'{weight_f1:.3f}', f'{int(sum(sup)):,}'])

    fig, ax = plt.subplots(figsize=(10, max(4, len(rows) * 0.55 + 2)))
    ax.axis('off')
    tbl = ax.table(
        cellText=rows,
        colLabels=['Класс', 'Precision', 'Recall', 'F1', 'Support'],
        cellLoc='center', loc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#1E293B'); cell.set_text_props(color='white', fontweight='bold')
        elif r > len(base_classes):
            cell.set_facecolor('#F1F5F9')
        elif r % 2 == 0:
            cell.set_facecolor('#F8FAFC')
        # Цвет строки по классу
        if 0 < r <= len(base_classes):
            cls = base_classes[r - 1]
            hex_col = COMP_COLORS.get(cls, '#94A3B8')
            cell.set_facecolor(hex_col + '33')  # прозрачный фон

    ax.set_title(f'Classification Report — soft labels (5 компонент)\n'
                 f'macro F1={macro_f1:.3f}  weighted F1={weight_f1:.3f}',
                 fontsize=12, fontweight='bold', pad=20)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  {out_path.name}')


# ─────────────────────────────────────────────────────────────────────────────
# Визуализация столба керна со стековыми барами
# ─────────────────────────────────────────────────────────────────────────────

def _load_csv_markup(well_name: str) -> list:
    """Загружает разметку геолога из CSV. Возвращает список {d_from, d_to, mineral, key}."""
    import csv
    csv_path = _cfg.PIPELINE_CSV / f'{well_name}.csv'
    if not csv_path.exists():
        return []
    rows = []
    with open(csv_path, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                d_from  = float(row['depth_from'])
                d_to    = float(row['depth_to'])
                mineral = row['mineral'].strip()
                key     = _csv_mineral_key(mineral)
                rows.append({'d_from': d_from, 'd_to': d_to, 'mineral': mineral, 'key': key})
            except (KeyError, ValueError):
                pass
    return sorted(rows, key=lambda r: r['d_from'])


def _draw_markup_strip(ax, markup: list, z_top: float, z_bot: float,
                        colormap: dict, title: str, show_text: bool = True):
    """Рисует полосу разметки с цветными блоками и текстом."""
    ax.set_xlim(0, 1); ax.set_ylim(z_bot, z_top)
    ax.set_title(title, fontsize=8, pad=3)
    ax.set_xticks([])

    for row in markup:
        y0 = max(row['d_from'], z_top)
        y1 = min(row['d_to'],   z_bot)
        if y0 >= y1:
            continue
        color = colormap.get(row['key'], _mineral_color(row['key']))
        ax.fill_betweenx([y0, y1], 0, 1, color=color, linewidth=0)
        ax.axhline(y0, color='white', linewidth=0.4, alpha=0.6)
        if show_text and (y1 - y0) > 0.05:
            mid = (y0 + y1) / 2
            label = row['mineral'][:22]
            ax.text(0.5, mid, label, ha='center', va='center',
                    fontsize=5.5, color='white', fontweight='bold',
                    wrap=True, clip_on=True)


def _draw_soft_strip(ax, segments: list, z_top: float, z_bot: float,
                     base_classes: list, soft_labels_map: dict = None,
                     title: str = '', show_entropy: bool = False):
    """Рисует стековые бары для soft-векторов. Если soft_labels_map — истинные метки."""
    ax.set_xlim(0, 1); ax.set_ylim(z_bot, z_top)
    ax.set_title(title, fontsize=8, pad=3)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(['0', '.5', '1'], fontsize=6)

    for seg in segments:
        y0, y1 = seg['depth_from'], seg['depth_to']
        probs   = seg['probs']
        left = 0.0
        for ci, (prob, cls) in enumerate(zip(probs, base_classes)):
            color = COMP_COLORS.get(cls, '#94A3B8')
            ax.fill_betweenx([y0, y1], left, left + prob, color=color, linewidth=0)
            if prob > 0.15:
                ax.text(left + prob / 2, (y0 + y1) / 2,
                        f'{prob:.0%}', ha='center', va='center',
                        fontsize=6, color='white', fontweight='bold')
            left += prob

    if show_entropy:
        # Энтропия справа как тонкая белая линия
        for seg in segments:
            y0, y1 = seg['depth_from'], seg['depth_to']
            probs = np.clip(seg['probs'], 1e-9, 1)
            H = -np.sum(probs * np.log2(probs)) / np.log2(len(probs))  # 0-1
            mid = (y0 + y1) / 2
            ax.plot([0.85, 0.85 + H * 0.13], [mid, mid],
                    color='white', linewidth=1.5, alpha=0.85, solid_capstyle='round')


@torch.no_grad()
def _soft_predict_column(img_ds: np.ndarray, img_uv: np.ndarray,
                         model, tfm_ds, tfm_uv,
                         depth_from: float, depth_to: float) -> list:
    """Прогоняет столб тайл за тайлом, возвращает список {depth_from, depth_to, probs[5]}."""
    h_px    = img_ds.shape[0]
    span_cm = (depth_to - depth_from) * 100
    tile_px = max(1, int(round(h_px / span_cm * _cfg.TILE_CM)))
    results, y, d = [], 0, depth_from
    while y + tile_px <= h_px:
        t_ds = tfm_ds(Image.fromarray(img_ds[y: y + tile_px])).unsqueeze(0).to(DEVICE)
        t_uv = tfm_uv(Image.fromarray(img_uv[y: y + tile_px])).unsqueeze(0).to(DEVICE)
        probs = F.softmax(model(t_ds, t_uv), dim=1).squeeze(0).cpu().numpy()
        results.append({
            'depth_from': d,
            'depth_to':   d + _cfg.TILE_CM / 100,
            'probs':      probs,
        })
        y += tile_px
        d += _cfg.TILE_CM / 100
    return results


def _parse_depth(fname: str):
    m = _DEPTH_RE.search(fname)
    if not m: return None
    return float(m.group(1).replace(',', '.')), float(m.group(2).replace(',', '.'))


def visualize_core_column(well_name: str, model, tfm_ds, tfm_uv,
                           base_classes: list, soft_labels_map: dict,
                           vis_dir: Path, target_depth=None):
    """Визуализирует столб керна: 6 панелей (фото + разметка + soft-истина + soft-предсказание)."""
    well_dir = _cfg.DIGITAL_CORE / well_name
    foto_dir = next((d for d in well_dir.iterdir()
                     if d.is_dir() and d.name.lower() == 'фото'), None)
    if foto_dir is None:
        print(f'  [SKIP] нет папки фото: {well_dir}')
        return

    ds_dir = foto_dir / 'ДС'
    uv_dir = foto_dir / 'УФ'
    if not ds_dir.exists():
        print(f'  [SKIP] нет ДС фото: {ds_dir}')
        return

    # Загружаем CSV-разметку геолога
    markup = _load_csv_markup(well_name)

    photos = []
    for ds_path in sorted(ds_dir.glob('*.jpeg')):
        d = _parse_depth(ds_path.stem)
        if d is None: continue
        d_from, d_to = d
        if target_depth is not None and abs(d_from - target_depth) > 0.05:
            continue
        uv_path = None
        if uv_dir.exists():
            uv_path = next(
                (p for p in uv_dir.glob('*.jpeg')
                 if abs((_parse_depth(p.stem) or (0, 0))[0] - d_from) < 0.05), None)
        if uv_path is None:
            continue
        photos.append((ds_path, uv_path, d_from, d_to))

    if not photos:
        print(f'  [SKIP] нет подходящих фото для {well_name}')
        return

    for ds_path, uv_path, d_from, d_to in photos:
        img_ds = np.array(Image.open(ds_path).convert('RGB'))
        img_uv = np.array(Image.open(uv_path).convert('RGB'))

        # Предсказания модели (тайл за тайлом)
        pred_segs = _soft_predict_column(
            img_ds, img_uv, model, tfm_ds, tfm_uv, d_from, d_to)
        if not pred_segs:
            continue

        z_top = pred_segs[0]['depth_from']
        z_bot = pred_segs[-1]['depth_to']
        h_cm  = (z_bot - z_top) * 100
        fig_h = max(10, int(h_cm * 0.28))

        # Истинные soft-метки для каждого тайла (из разметки CSV)
        true_segs = []
        for seg in pred_segs:
            mid = (seg['depth_from'] + seg['depth_to']) / 2
            # Находим интервал в CSV
            matched = [r for r in markup if r['d_from'] <= mid <= r['d_to']]
            if matched:
                key = matched[0]['key']
                soft_vec = soft_labels_map.get(key)
                if soft_vec is None:
                    # fallback: ищем без учёта частичного совпадения
                    soft_vec = [0.2] * len(base_classes)
            else:
                soft_vec = [1.0 / len(base_classes)] * len(base_classes)
            true_segs.append({**seg, 'probs': np.array(soft_vec)})

        # 6 колонок: ДС | УФ | Геолог | 6-класс | Soft-истина | Soft-предсказание
        fig, axes = plt.subplots(
            1, 6, figsize=(20, fig_h),
            gridspec_kw={'width_ratios': [1.2, 1.2, 1.4, 1.4, 2.0, 2.0]})

        # ── ДС фото ──────────────────────────────────────────────────────────
        for ax, img, title in zip(axes[:2], [img_ds, img_uv],
                                   ['ДС (дневной)', 'УФ']):
            ax.imshow(img, aspect='auto', extent=[0, 1, z_bot, z_top])
            ax.set_title(title, fontsize=8, pad=3)
            ax.set_xlim(0, 1); ax.set_ylim(z_bot, z_top)
            ax.set_xticks([])
            ax.set_ylabel('Глубина (м)', fontsize=7)
            ax.yaxis.set_tick_params(labelsize=7)

        # ── Геолог (CSV) ─────────────────────────────────────────────────────
        vis_markup = [r for r in markup
                      if r['d_to'] > z_top - 0.05 and r['d_from'] < z_bot + 0.05]
        _draw_markup_strip(axes[2], vis_markup, z_top, z_bot,
                           colormap={}, title='Геолог\n(оригинал)',
                           show_text=True)
        axes[2].set_ylabel('')
        axes[2].yaxis.set_tick_params(labelsize=7)

        # ── 6-класс маппинг ──────────────────────────────────────────────────
        mapped_markup = []
        for r in vis_markup:
            cls6 = MINERAL_TO_6CLASS.get(r['key'], 'unknown')
            mapped_markup.append({**r, 'mineral': cls6, 'key': cls6})
        _draw_markup_strip(axes[3], mapped_markup, z_top, z_bot,
                           colormap=SIX_CLASS_COLORS, title='6-классов\n(маппинг)',
                           show_text=True)
        axes[3].yaxis.set_tick_params(labelsize=7)

        # ── Soft-истина (из CSV + SOFT_LABEL_MAP) ────────────────────────────
        _draw_soft_strip(axes[4], true_segs, z_top, z_bot, base_classes,
                         title='Soft-истина\n(из разметки)')
        axes[4].yaxis.set_tick_params(labelsize=7)

        # ── Soft-предсказание модели (+ энтропия) ────────────────────────────
        _draw_soft_strip(axes[5], pred_segs, z_top, z_bot, base_classes,
                         title='Soft-предсказание\n(модель  |  ▬ энтропия)',
                         show_entropy=True)
        axes[5].yaxis.set_tick_params(labelsize=7)

        # ── Легенда компонент ─────────────────────────────────────────────────
        comp_patches = [mpatches.Patch(color=COMP_COLORS.get(c, '#94A3B8'),
                                       label=COMP_SHORT.get(c, c))
                        for c in base_classes]
        axes[5].legend(handles=comp_patches, loc='lower right', fontsize=7,
                       framealpha=0.85, ncol=1, title='Компоненты', title_fontsize=7)

        # ── Легенда геолога (уникальные минералы) ────────────────────────────
        seen_minerals = {}
        for r in vis_markup:
            if r['key'] not in seen_minerals:
                seen_minerals[r['key']] = r['mineral'][:25]
        geo_patches = [mpatches.Patch(color=_mineral_color(k), label=v[:22])
                       for k, v in list(seen_minerals.items())[:8]]
        if geo_patches:
            axes[2].legend(handles=geo_patches, loc='lower left', fontsize=5.5,
                           framealpha=0.85, ncol=1, title='Минерал', title_fontsize=6)

        # ── Легенда 6-классов ─────────────────────────────────────────────────
        seen_6 = set(r.get('key', 'unknown') for r in mapped_markup)
        six_patches = [mpatches.Patch(color=SIX_CLASS_COLORS.get(c, '#94A3B8'), label=c)
                       for c in seen_6 if c in SIX_CLASS_COLORS]
        if six_patches:
            axes[3].legend(handles=six_patches, loc='lower left', fontsize=6,
                           framealpha=0.85, ncol=1, title='6-классов', title_fontsize=6)

        fig.suptitle(
            f'{well_name}  {d_from:.2f}–{d_to:.2f} м  ({h_cm:.0f} см)\n'
            f'Геолог → 6-классов → Soft-истина → Soft-предсказание',
            fontsize=11, fontweight='bold', y=1.01)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        safe = well_name.replace(' ', '_')
        out = vis_dir / f'{safe}_{d_from:.2f}-{d_to:.2f}_soft.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Сохранено: {out.name}')


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run',    default='run_soft_dual',
                   help='Папка в results/ (default: run_soft_dual)')
    p.add_argument('--split',  default='test', choices=['train', 'val', 'test'])
    p.add_argument('--no-columns', action='store_true',
                   help='Пропустить визуализацию столбов керна')
    p.add_argument('--well',   default=None, help='Одна скважина для столба')
    p.add_argument('--depth',  type=float, default=None, help='depth_from для --well')
    return p.parse_args()


def main():
    args   = parse_args()
    run_dir = _cfg.RESULTS_ROOT / args.run
    dataset_root = Path('data/dataset_soft')

    vis_dir   = run_dir / 'visual'
    stats_dir = vis_dir / 'stats'
    vis_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    # Загружаем метаданные датасета
    with open(dataset_root / 'normalization_stats.json') as f:
        stats = json.load(f)
    with open(dataset_root / 'soft_labels.json') as f:
        soft_labels = json.load(f)
    with open(dataset_root / 'base_classes.json') as f:
        base_classes = json.load(f)

    ds_mean = stats['ДС']['mean']; ds_std = stats['ДС']['std']
    uv_mean = stats['УФ']['mean']; uv_std = stats['УФ']['std']

    print(f'Run          : {run_dir}')
    print(f'Dataset      : {dataset_root}')
    print(f'Split        : {args.split}')
    print(f'Базовые      : {base_classes}')
    print(f'Device       : {DEVICE}')

    # Загружаем модель
    model = load_model(run_dir, num_classes=len(base_classes))
    print('Модель загружена.')

    # ── Инференс на сплите ────────────────────────────────────────────────────
    print(f'\n[1/5] Инференс на {args.split}-сете...')
    probs_arr, true_soft_arr, true_hard, pred_hard, orig_classes = collect_test_predictions(
        model, dataset_root, args.split, soft_labels, base_classes,
        ds_mean, ds_std, uv_mean, uv_std,
    )
    n = len(true_hard)
    macro_f1 = f1_score(true_hard, pred_hard, average='macro', zero_division=0)
    acc      = (true_hard == pred_hard).mean()
    print(f'  n={n:,}  macro F1={macro_f1:.4f}  Acc={acc:.4f}')
    print(f'\n{classification_report(true_hard, pred_hard, target_names=base_classes, zero_division=0)}')

    # ── Графики ───────────────────────────────────────────────────────────────
    print('[2/5] Матрица ошибок...')
    plot_confusion(true_hard, pred_hard, base_classes, stats_dir / 'confusion_soft.png')

    print('[3/5] F1 по компонентам...')
    plot_f1_bars(true_hard, pred_hard, base_classes, stats_dir / 'f1_components.png')

    print('[4/5] Средний состав по классу...')
    plot_avg_composition(probs_arr, true_hard, base_classes,
                         stats_dir / 'avg_composition.png')

    print('[4b] Уверенность...')
    plot_confidence(probs_arr, true_hard, pred_hard, stats_dir / 'confidence_soft.png')

    print('[4c] Отчёт (таблица)...')
    plot_report_table(true_hard, pred_hard, base_classes,
                      stats_dir / 'classification_report_soft.png')

    print('[4d] Энтропия по классу...')
    plot_entropy_by_class(probs_arr, true_hard, base_classes,
                          stats_dir / 'entropy_by_class.png')

    print('[4e] KL-дивергенция...')
    plot_kl_divergence(probs_arr, true_soft_arr, true_hard, base_classes,
                       stats_dir / 'kl_divergence.png')

    # ── Столбы керна ─────────────────────────────────────────────────────────
    if not args.no_columns:
        print('\n[5/5] Визуализация столбов керна...')
        tfm_ds = get_transforms('square', 'none', False, ds_mean, ds_std)
        tfm_uv = get_transforms('square', 'none', False, uv_mean, uv_std)

        if args.well:
            wells_to_show = [(args.well, args.depth)]
        else:
            wells_to_show = [
                ('Харасавэйск_700',  800.00),
                ('Харасавэйск_300',  797.80),
                ('Харасавэйск_1800', 648.00),
            ]

        for well_name, depth in wells_to_show:
            print(f'  {well_name} depth≈{depth}...')
            visualize_core_column(
                well_name, model, tfm_ds, tfm_uv,
                base_classes, soft_labels, vis_dir, target_depth=depth,
            )

    print(f'\nГотово! Визуализация: {vis_dir}')
    print(f'  macro F1 = {macro_f1:.4f}')


if __name__ == '__main__':
    main()
