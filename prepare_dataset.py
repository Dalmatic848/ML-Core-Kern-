"""
Единый пайплайн подготовки данных.

Раньше это были два отдельных скрипта (prepare_data.py + prepare_exp.py) с
собственной копией маппинга минерал→класс в каждом. Теперь один вход,
таксономия — из src/taxonomy.py.

Два независимых блока (сознательно НЕ объединены в один алгоритм —
см. docstring build_default_dataset()/build_variant_dataset()):

  1. Тайлинг (--stages 1,2)         — Excel→CSV, нарезка фото на тайлы.
     Нужен один раз; дальше все варианты датасета строятся поверх уже
     готовых тайлов в data/pipeline/tiles/.

  2. Сборка варианта датасета (--variant ...):
     default        — 6 классов (слияние), сплит по скважинам. Канонический
                       датасет, на котором построены все 17 экспериментов.
     random         — 6 классов (слияние), случайный сплит по интервалам.
     no-merge       — оригинальные минералы (без слияния), случайный сплит.
     no-merge-well  — оригинальные минералы, сплит по скважинам.
     soft           — мягкие метки (5 базовых компонент), сплит по
                      интервалам (--split-mode well тоже поддерживается).

Примеры:
    python3 prepare_dataset.py --force                       # канонический датасет с нуля
    python3 prepare_dataset.py --stages 3,4,5,6,7             # пересобрать канонический без тайлинга
    python3 prepare_dataset.py --variant random
    python3 prepare_dataset.py --variant no-merge --min-tiles 300
    python3 prepare_dataset.py --variant soft
"""

import argparse
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import config as _cfg
from src.taxonomy import (
    MINERAL_TO_CLASS,
    load_soft_labels_config,
    mineral_to_class,
    normalize_mineral_name,
)

IMG_EXTS   = {'.jpg', '.jpeg', '.png', '.bmp'}
TILE_MIN_H = 20
_DEPTH_RE  = re.compile(r'_([\d]+[.,][\d]+)\s*-\s*([\d]+[.,][\d]+)$')
FRAG_RE    = re.compile(r'_(\d+\.\d{3})_(\d+\.\d{3})_frag\d{4}\.jpg$')
MODALITIES = _cfg.MODALITIES


def _load(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2BGR)


def _save(path: Path, img: np.ndarray) -> None:
    rgb = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(str(path), quality=90)


def _parse_depth(stem: str) -> Optional[Tuple[float, float]]:
    m = _DEPTH_RE.search(stem)
    if not m:
        return None
    return float(m.group(1).replace(',', '.')), float(m.group(2).replace(',', '.'))


# ═════════════════════════════════════════════════════════════════════════════
# БЛОК 1 — Тайлинг (Excel → CSV → тайлы). Общий для всех вариантов датасета.
# ═════════════════════════════════════════════════════════════════════════════

def stage1_excel_to_csv(data_root: Path, csv_root: Path) -> None:
    csv_root.mkdir(parents=True, exist_ok=True)
    wells = sorted(d for d in data_root.iterdir() if d.is_dir())
    print(f'Stage 1 — Excel → CSV  ({len(wells)} скважин)')
    ok = skip = fail = 0
    for well_dir in wells:
        csv_path = csv_root / f'{well_dir.name}.csv'
        if csv_path.exists():
            skip += 1
            continue
        xlsx = [f for f in well_dir.glob('*.xlsx') if not f.name.startswith('~$')]
        if not xlsx:
            fail += 1
            continue
        try:
            raw = pd.read_excel(xlsx[0], usecols='B:G', skiprows=8, header=None)
            raw = raw.drop(5, axis=1)
            for col in [1, 3, 4]:
                raw[col] = pd.to_numeric(raw[col], errors='coerce')
            raw = raw.dropna(subset=[1, 3, 4])
            raw[2] = raw[1] + raw[4]
            raw[1] = raw[1] + raw[3]
            raw = raw.drop([3, 4], axis=1).dropna()
            raw = raw.rename(columns={1: 'depth_from', 2: 'depth_to', 6: 'mineral'})
            raw['mineral'] = raw['mineral'].astype(str).str.strip()
            raw.to_csv(csv_path, index=False, encoding='utf-8-sig')
            ok += 1
        except Exception as e:
            print(f'  [ERR] {well_dir.name}: {e}')
            fail += 1
    print(f'  создано={ok}  кеш={skip}  ошибок={fail}')


def _process_well(well_dir: Path, csv_path: Path, out_root: Path,
                  modalities: List[str], tile_cm: float, overlap_cm: float) -> Tuple[int, set]:
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    foto_dir = next((c for c in well_dir.iterdir() if c.is_dir() and c.name.lower() == 'фото'), None)
    if foto_dir is None:
        return 0, set()

    total = 0
    unknown_set: set = set()

    for modality in modalities:
        mod_dir = foto_dir / modality
        if not mod_dir.exists():
            continue

        photos: List[Tuple[float, float, Path]] = []
        for img_path in sorted(mod_dir.iterdir()):
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            depths = _parse_depth(img_path.stem)
            if depths:
                photos.append((depths[0], depths[1], img_path))

        if not photos:
            continue

        for _, row in df.iterrows():
            mineral_key = normalize_mineral_name(str(row['mineral']))
            if mineral_key not in MINERAL_TO_CLASS:
                unknown_set.add(mineral_key)
                continue
            mapped_class = MINERAL_TO_CLASS[mineral_key]
            if mapped_class is None:
                continue

            d_from = float(row['depth_from'])
            d_to   = float(row['depth_to'])
            if d_to <= d_from:
                continue
            total_cm = (d_to - d_from) * 100.0
            if total_cm < tile_cm:
                continue

            overlaps = [(pf, pt, pp) for pf, pt, pp in photos if pf < d_to and pt > d_from]
            if not overlaps:
                continue

            crops: List[np.ndarray] = []
            for pf, pt, pp in sorted(overlaps, key=lambda x: x[0]):
                inter_s = max(d_from, pf)
                inter_e = min(d_to, pt)
                if inter_s >= inter_e:
                    continue
                try:
                    img = _load(pp)
                except Exception:
                    continue
                h = img.shape[0]
                ph_cm = (pt - pf) * 100.0
                if ph_cm <= 0:
                    continue
                ppc = h / ph_cm
                y0 = max(0, int((inter_s - pf) * 100.0 * ppc))
                y1 = min(h, int(round((inter_e - pf) * 100.0 * ppc)))
                if y0 < y1:
                    crops.append(img[y0:y1, :])

            if not crops:
                continue

            max_w  = max(c.shape[1] for c in crops)
            padded = [cv2.copyMakeBorder(c, 0, 0, 0, max_w - c.shape[1], cv2.BORDER_CONSTANT, value=0)
                      if c.shape[1] < max_w else c for c in crops]
            merged = np.vstack(padded)

            h_total = merged.shape[0]
            ppc     = h_total / total_cm
            tile_px = max(1, int(round(tile_cm * ppc)))

            dst_dir = out_root / well_dir.name / modality / mapped_class
            dst_dir.mkdir(parents=True, exist_ok=True)

            frag = 0
            cur_cm = 0.0
            while cur_cm + tile_cm <= total_cm + 1e-6:
                y0 = int(round(cur_cm * ppc))
                y1 = y0 + tile_px
                if y1 > h_total:
                    break
                tile = merged[y0:y1, :]
                if tile.shape[0] >= TILE_MIN_H:
                    fname = f'{mineral_key}_{d_from:.3f}_{d_to:.3f}_frag{frag:04d}.jpg'
                    _save(dst_dir / fname, tile)
                    frag  += 1
                    total += 1
                cur_cm += (tile_cm - overlap_cm)

    return total, unknown_set


def stage2_tile(data_root: Path, csv_root: Path, tile_root: Path,
                modalities: List[str], tile_cm: float, overlap_cm: float,
                force: bool) -> None:
    """Нарезает фото на тайлы. Директория тайла = слитый 6-классовый таксон
    (для канонического датасета), но имя файла хранит СЫРОЕ название минерала
    (`{mineral}_{depth_from}_{depth_to}_frag####.jpg`) — это то, что позволяет
    build_variant_dataset() ниже восстанавливать оригинальную минералогию из
    уже нарезанных тайлов, не трогая фото повторно."""
    if force and tile_root.exists():
        print(f'Stage 2 — удаляем {tile_root}')
        shutil.rmtree(tile_root)
    tile_root.mkdir(parents=True, exist_ok=True)

    wells = sorted(d for d in data_root.iterdir() if d.is_dir())
    print(f'Stage 2 — Тайлинг ({len(wells)} скважин, tile={tile_cm} cm, overlap={overlap_cm} cm)')
    grand_total = 0
    all_unknown: set = set()
    for well_dir in tqdm(wells, desc='  скважины'):
        csv_path = csv_root / f'{well_dir.name}.csv'
        if not csv_path.exists():
            continue
        n, unk = _process_well(well_dir, csv_path, tile_root, modalities, tile_cm, overlap_cm)
        grand_total += n
        if unk:
            print(f'  [WARN] {well_dir.name}: неизвестные минералы: {unk}')
            all_unknown |= unk
    print(f'  всего тайлов: {grand_total:,}')
    if all_unknown:
        print(f'  добавьте в src/taxonomy.py MINERAL_TO_CLASS: {sorted(all_unknown)}')


# ═════════════════════════════════════════════════════════════════════════════
# БЛОК 2а — Канонический датасет: 6 классов, сплит по скважинам.
#
# Это ЕДИНСТВЕННЫЙ вариант, где сплит устроен как раздельная жадная
# оптимизация по (скважина → train/val/test), учитывающая все 6 классов
# одновременно (MSE отклонения от целевой доли + штраф за дефицит).
# Это тот код, на котором обучены все 17 опубликованных экспериментов —
# сознательно НЕ трогаем его при объединении с prepare_exp.py (см. модуль
# docstring и docstring build_variant_dataset ниже: там сплит устроен
# иначе, по интервалам, потому что raw-минералы физически не видны в
# структуре директорий тайлов — только в имени файла).
# ═════════════════════════════════════════════════════════════════════════════

def scan_tiles_by_directory(tile_root: Path, classes_order: List[str]) -> Dict:
    counts: Dict = defaultdict(lambda: defaultdict(Counter))
    for well_dir in sorted(tile_root.iterdir()):
        if not well_dir.is_dir():
            continue
        for mod_dir in well_dir.iterdir():
            if not mod_dir.is_dir():
                continue
            for cls_dir in mod_dir.iterdir():
                if not cls_dir.is_dir():
                    continue
                n = sum(1 for f in cls_dir.iterdir() if f.suffix.lower() == '.jpg')
                if n > 0:
                    counts[well_dir.name][mod_dir.name][cls_dir.name] = n
    wcc = dict(counts)

    class_totals: Counter = Counter()
    for mods in wcc.values():
        for cls_counts in mods.values():
            class_totals.update(cls_counts)

    print(f'\nStage 3 — Статистика тайлов (скважин={len(wcc)}):')
    print(f'  {"Класс":<52} {"Итого":>8}')
    print('  ' + '-' * 62)
    for cls in classes_order:
        print(f'  {cls:<52} {class_totals.get(cls, 0):>8,}')
    unknown_cls = set(class_totals) - set(classes_order)
    if unknown_cls:
        print(f'  Классы вне CLASSES_ORDER: {unknown_cls}')

    return wcc


def _well_totals(wcc: Dict) -> Dict[str, Counter]:
    result = {}
    for well, mods in wcc.items():
        c: Counter = Counter()
        for cls_counts in mods.values():
            c.update(cls_counts)
        result[well] = c
    return result


def split_well_greedy(wcc: Dict, classes_order: List[str],
                      val_frac: float, test_frac: float, seed: int,
                      min_class_tiles: int) -> Tuple[Dict[str, str], Dict[str, Counter]]:
    rng = random.Random(seed)
    wt  = _well_totals(wcc)
    global_total: Counter = Counter()
    for c in wt.values():
        global_total.update(c)

    all_classes = [c for c in classes_order if global_total.get(c, 0) > 0]
    wells = list(wt.keys())
    rng.shuffle(wells)

    split_counts: Dict[str, Counter] = {'train': Counter(), 'val': Counter(), 'test': Counter()}
    well_to_split: Dict[str, str] = {}
    unassigned = list(wells)

    def _effective_min(cls: str) -> int:
        return min(min_class_tiles, max(1, global_total.get(cls, 1) // 4))

    def _score(w: str, target_split: str, target_frac: float) -> float:
        trial = Counter(split_counts[target_split])
        trial.update(wt[w])
        mse = sum((trial[c] / max(global_total[c], 1) - target_frac) ** 2 for c in all_classes)
        deficit = sum(max(0, _effective_min(c) - trial[c]) ** 2 for c in all_classes)
        return mse + 1e-3 * deficit

    def _assign(target_split: str, target_frac: float) -> None:
        while unassigned:
            assigned = sum(split_counts[target_split].values())
            grand    = sum(global_total.values())
            frac_ok  = (grand > 0 and assigned / grand >= target_frac)
            min_ok   = all(split_counts[target_split].get(c, 0) >= _effective_min(c)
                           for c in all_classes)
            if frac_ok and min_ok:
                break
            best = min(unassigned, key=lambda w: _score(w, target_split, target_frac))
            unassigned.remove(best)
            well_to_split[best] = target_split
            split_counts[target_split].update(wt[best])

    _assign('val',  val_frac)
    _assign('test', test_frac)
    for w in unassigned:
        well_to_split[w] = 'train'
        split_counts['train'].update(wt[w])

    print('\nStage 4 — Сплит по скважинам:')
    for split in ['train', 'val', 'test']:
        ws = sorted(w for w, s in well_to_split.items() if s == split)
        print(f'  {split:5}: {ws}')
    print(f'\n  {"Класс":<20} {"Train":>8} {"Val":>6} {"Test":>6} {"Val%":>6} {"Test%":>6}')
    print('  ' + '-' * 55)
    for cls in classes_order:
        tr  = split_counts['train'].get(cls, 0)
        vl  = split_counts['val'].get(cls, 0)
        te  = split_counts['test'].get(cls, 0)
        tot = tr + vl + te
        vp  = vl / tot * 100 if tot else 0
        tp  = te / tot * 100 if tot else 0
        flag = '  !' if (vl < min_class_tiles or te < min_class_tiles) else ''
        print(f'  {cls:<20} {tr:>8,} {vl:>6,} {te:>6,} {vp:>5.1f}% {tp:>5.1f}%{flag}')

    return well_to_split, split_counts


def copy_well_split_to_dataset(tile_root: Path, ds_root: Path, well_to_split: Dict[str, str],
                               modalities: List[str], classes_order: List[str], force: bool) -> None:
    if force and ds_root.exists():
        print(f'Stage 5 — удаляем {ds_root}')
        shutil.rmtree(ds_root)
    ds_root.mkdir(parents=True, exist_ok=True)

    total_copied = 0
    manifest: Dict = {}
    for well, split in tqdm(well_to_split.items(), desc='Stage 5 — копирование'):
        manifest[well] = {'split': split, 'classes': {}}
        well_dir = tile_root / well
        if not well_dir.exists():
            continue
        for mod_dir in well_dir.iterdir():
            if not mod_dir.is_dir() or mod_dir.name not in modalities:
                continue
            for cls_dir in mod_dir.iterdir():
                if not cls_dir.is_dir():
                    continue
                dst = ds_root / mod_dir.name / split / cls_dir.name
                dst.mkdir(parents=True, exist_ok=True)
                n = 0
                for src_file in cls_dir.iterdir():
                    if src_file.suffix.lower() == '.jpg':
                        shutil.copy2(src_file, dst / src_file.name)
                        n += 1
                        total_copied += 1
                key = f'{mod_dir.name}/{cls_dir.name}'
                manifest[well]['classes'][key] = manifest[well]['classes'].get(key, 0) + n

    class_to_idx = {cls: idx for idx, cls in enumerate(classes_order)}
    with open(ds_root / 'label_encoder.json', 'w', encoding='utf-8') as f:
        json.dump(class_to_idx, f, ensure_ascii=False, indent=2)
    with open(ds_root / 'split_manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f'  скопировано: {total_copied:,} файлов → {ds_root}')


def stage6_final_stats(ds_root: Path, modalities: List[str], classes_order: List[str]) -> Dict:
    counts: Dict = defaultdict(lambda: defaultdict(Counter))
    for mod in modalities:
        mod_dir = ds_root / mod
        if not mod_dir.exists():
            continue
        for split_dir in mod_dir.iterdir():
            if not split_dir.is_dir():
                continue
            for cls_dir in split_dir.iterdir():
                if not cls_dir.is_dir():
                    continue
                n = sum(1 for f in cls_dir.iterdir() if f.suffix.lower() == '.jpg')
                counts[mod][split_dir.name][cls_dir.name] = n
    stats = dict(counts)

    for mod in modalities:
        mst = stats.get(mod, {})
        print(f'\nStage 6 — {mod}:')
        print(f'  {"Класс":<52} {"Train":>8} {"Val":>6} {"Test":>6}')
        print('  ' + '-' * 70)
        for cls in classes_order:
            tr = mst.get('train', {}).get(cls, 0)
            vl = mst.get('val',   {}).get(cls, 0)
            te = mst.get('test',  {}).get(cls, 0)
            print(f'  {cls:<52} {tr:>8,} {vl:>6,} {te:>6,}')
        tr_t = sum(mst.get('train', {}).values())
        vl_t = sum(mst.get('val',   {}).values())
        te_t = sum(mst.get('test',  {}).values())
        print(f'  {"ИТОГО":<52} {tr_t:>8,} {vl_t:>6,} {te_t:>6,}')
        tr_vals = [v for v in mst.get('train', {}).values() if v > 0]
        if tr_vals:
            print(f'  дисбаланс train: {max(tr_vals)/max(min(tr_vals),1):.1f}:1')
    return stats


def stage7_normalization(ds_root: Path, modalities: List[str], norm_stats_file: Path,
                         sample_size: int, seed: int) -> Dict:
    rng = random.Random(seed)
    stats: Dict = {}
    print(f'\nStage 7 — Нормализация (sample={sample_size} на модальность):')
    for mod in modalities:
        train_dir = ds_root / mod / 'train'
        all_imgs  = list(train_dir.rglob('*.jpg'))
        sample    = rng.sample(all_imgs, min(sample_size, len(all_imgs)))
        sum_mean  = np.zeros(3, dtype=np.float64)
        sum_std   = np.zeros(3, dtype=np.float64)
        for img_path in tqdm(sample, desc=f'  {mod}', leave=False):
            img = np.array(Image.open(img_path).convert('RGB')).astype(np.float32) / 255.0
            sum_mean += img.mean(axis=(0, 1))
            sum_std  += img.std(axis=(0, 1))
        n    = len(sample)
        mean = [round(float(v), 4) for v in sum_mean / n]
        std  = [round(float(v), 4) for v in sum_std  / n]
        stats[mod] = {'mean': mean, 'std': std}
        print(f'  {mod}: mean={mean}  std={std}')
    with open(norm_stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f'  сохранено: {norm_stats_file}')
    return stats


def stage8_visual_artifacts(ds_root: Path, classes_order: List[str], modalities: List[str],
                            dataset_stats: Path, tile_cm: float, seed: int) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    from config import CLASS_PALETTE, CLASS_SHORT

    dataset_stats.mkdir(parents=True, exist_ok=True)
    mst = {}
    for mod in modalities:
        mst[mod] = {}
        mod_dir = ds_root / mod
        if not mod_dir.exists():
            continue
        for split_dir in mod_dir.iterdir():
            if split_dir.is_dir():
                mst[mod][split_dir.name] = {}
                for cls_dir in split_dir.iterdir():
                    if cls_dir.is_dir():
                        mst[mod][split_dir.name][cls_dir.name] = sum(
                            1 for f in cls_dir.iterdir() if f.suffix.lower() == '.jpg')

    first_mod = modalities[0]
    m = mst.get(first_mod, {})
    splits = ['train', 'val', 'test']
    split_colors = {'train': '#3B82F6', 'val': '#F59E0B', 'test': '#EF4444'}
    short_names  = [CLASS_SHORT.get(c, c[:10]) for c in classes_order]

    fig, ax = plt.subplots(figsize=(14, 5))
    n_cls = len(classes_order)
    x = np.arange(n_cls)
    width = 0.25
    for i, split in enumerate(splits):
        counts = [m.get(split, {}).get(c, 0) for c in classes_order]
        bars = ax.bar(x + i * width, counts, width, label=split.capitalize(),
                      color=split_colors[split], edgecolor='#333', linewidth=0.5)
        for bar, cnt in zip(bars, counts):
            if cnt > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                        f'{cnt:,}', ha='center', va='bottom', fontsize=7, rotation=90)
    ax.set_xticks(x + width)
    ax.set_xticklabels(short_names, rotation=30, ha='right', fontsize=10)
    ax.set_ylabel('Количество тайлов')
    ax.set_title(f'Распределение классов (тайл {tile_cm} см, {first_mod})', fontsize=12)
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{int(v):,}'))
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    fig.savefig(dataset_stats / 'class_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

    rng = random.Random(seed)
    n_examples = 4
    fig, axes = plt.subplots(n_cls, n_examples, figsize=(n_examples * 2.2, n_cls * 2.8))
    if n_cls == 1:
        axes = [axes]
    fig.suptitle(f'Примеры тайлов ({tile_cm} см, {first_mod})', fontsize=13, fontweight='bold', y=1.01)
    for row_i, cls in enumerate(classes_order):
        cls_dir = ds_root / first_mod / 'train' / cls
        if not cls_dir.exists():
            cls_dir = ds_root / first_mod / 'val' / cls
        files  = list(cls_dir.glob('*.jpg')) if cls_dir.exists() else []
        sample = rng.sample(files, min(n_examples, len(files)))
        color  = CLASS_PALETTE.get(cls, '#94A3B8')
        short  = CLASS_SHORT.get(cls, cls[:10])
        for col_i in range(n_examples):
            ax = axes[row_i][col_i] if hasattr(axes[row_i], '__iter__') else axes[row_i]
            if col_i < len(sample):
                ax.imshow(np.array(Image.open(sample[col_i]).convert('RGB')))
            ax.axis('off')
            if col_i == 0:
                ax.set_ylabel(short, rotation=0, labelpad=55, va='center',
                              fontsize=10, fontweight='bold', color=color)
    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    fig.savefig(dataset_stats / 'tile_examples.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nStage 8 — артефакты сохранены в: {dataset_stats}')


def build_default_dataset(args, stages: set) -> None:
    """Канонический датасет (см. модуль docstring). Точное поведение бывшего
    prepare_data.py — единственное отличие: MINERAL_TO_CLASS теперь
    импортируется из src/taxonomy.py вместо локальной копии CLASS_RULES."""
    tile_cm        = args.tile_cm   or _cfg.TILE_CM
    overlap_cm     = args.overlap_cm if args.overlap_cm is not None else _cfg.OVERLAP_CM
    min_tiles      = args.min_tiles or _cfg.MIN_CLASS_TILES
    classes_order  = _cfg.CLASSES_ORDER
    modalities     = _cfg.MODALITIES

    data_root      = _cfg.DIGITAL_CORE
    csv_root       = _cfg.PIPELINE_CSV
    tile_root      = _cfg.PIPELINE_TILES
    ds_root        = _cfg.DATASET_ROOT
    norm_stats     = _cfg.NORM_STATS_FILE
    dataset_stats  = _cfg.DATASET_STATS

    print('Вариант       : default (канонический)')
    print(f'CLASSES_ORDER : {classes_order}')
    print(f'tile_cm       : {tile_cm} см  |  overlap_cm: {overlap_cm} см')
    print(f'min_tiles     : {min_tiles}')
    print(f'Stages        : {sorted(stages)}')
    print()

    if 1 in stages:
        stage1_excel_to_csv(data_root, csv_root)
    if 2 in stages:
        stage2_tile(data_root, csv_root, tile_root, modalities, tile_cm, overlap_cm, args.force)

    wcc = None
    if 3 in stages or 4 in stages:
        wcc = scan_tiles_by_directory(tile_root, classes_order)

    well_to_split = None
    if 4 in stages:
        well_to_split, _ = split_well_greedy(wcc, classes_order,
                                             _cfg.VAL_FRAC, _cfg.TEST_FRAC, _cfg.SEED, min_tiles)

    if 5 in stages:
        if well_to_split is None:
            raise RuntimeError('Stage 5 требует Stage 4 (добавьте 4 в --stages)')
        copy_well_split_to_dataset(tile_root, ds_root, well_to_split, modalities, classes_order, args.force)

    if 6 in stages:
        stage6_final_stats(ds_root, modalities, classes_order)
    if 7 in stages:
        stage7_normalization(ds_root, modalities, norm_stats, args.sample_size, _cfg.SEED)
    if 8 in stages and not args.no_visuals:
        stage8_visual_artifacts(ds_root, classes_order, modalities, dataset_stats, tile_cm, _cfg.SEED)

    print('\nГотово.')


# ═════════════════════════════════════════════════════════════════════════════
# БЛОК 2б — Альтернативные варианты: random / no-merge / no-merge-well / soft.
#
# Работают ПОВЕРХ уже готовых тайлов (data/pipeline/tiles/), восстанавливая
# сырое название минерала из ИМЕНИ ФАЙЛА (FRAG_RE) — директории тайлов хранят
# только слитый 6-классовый таксон, поэтому этот блок не может переиспользовать
# scan_tiles_by_directory()/split_well_greedy() из БЛОКА 2а.
# ═════════════════════════════════════════════════════════════════════════════

def scan_tiles_by_interval(tile_root: Path) -> dict:
    """dict: interval_uid -> {well, mineral, files: {mod: [path, ...]}}.
    interval_uid = '{well}||{mineral}_{d_from}_{d_to}'."""
    intervals: dict = {}
    tile_root = Path(tile_root)

    for well_dir in sorted(tile_root.iterdir()):
        if not well_dir.is_dir():
            continue
        for mod_dir in well_dir.iterdir():
            if not mod_dir.is_dir() or mod_dir.name not in MODALITIES:
                continue
            for cls_dir in mod_dir.iterdir():
                if not cls_dir.is_dir():
                    continue
                for f in cls_dir.iterdir():
                    if f.suffix.lower() != '.jpg':
                        continue
                    m = FRAG_RE.search(f.name)
                    if not m:
                        continue
                    mineral = f.name[:m.start()]
                    interval_key = f'{mineral}_{m.group(1)}_{m.group(2)}'
                    uid = f'{well_dir.name}||{interval_key}'
                    if uid not in intervals:
                        intervals[uid] = {
                            'well': well_dir.name,
                            'mineral': mineral,
                            'files': defaultdict(list),
                        }
                    intervals[uid]['files'][mod_dir.name].append(f)

    return intervals


def assign_classes(intervals: dict, no_merge: bool) -> dict:
    skipped = 0
    for uid, info in intervals.items():
        mineral = info['mineral']
        if no_merge:
            info['class_name'] = mineral
        else:
            mapped = mineral_to_class(mineral)
            if mapped is None:
                info['class_name'] = None
                skipped += 1
            else:
                info['class_name'] = mapped
    if skipped:
        print(f'  [skip] {skipped} интервалов с неизвестным/удалённым минералом')
    return intervals


def filter_classes(intervals: dict, min_tiles: int) -> tuple:
    class_tiles: Counter = Counter()
    for info in intervals.values():
        if info['class_name'] is None:
            continue
        n = len(info['files'].get('ДС', info['files'].get(list(info['files'].keys())[0], [])))
        class_tiles[info['class_name']] += n

    print('\n=== Тайлов по классам (ДС) ===')
    kept_classes = set()
    for cls, n in sorted(class_tiles.items(), key=lambda x: -x[1]):
        flag = '  ✓' if n >= min_tiles else f'  ✗ (< {min_tiles}, удаляем)'
        print(f'  {cls:<55} {n:>5}{flag}')
        if n >= min_tiles:
            kept_classes.add(cls)

    print(f'\nОставляем {len(kept_classes)} классов из {len(class_tiles)}')
    filtered = {uid: info for uid, info in intervals.items()
                if info['class_name'] in kept_classes}
    return filtered, sorted(kept_classes)


def split_random(intervals: dict, classes_order: list,
                 val_frac: float = 0.10, test_frac: float = 0.10,
                 seed: int = 42) -> dict:
    """Стратифицированный случайный сплит по интервалам (не по тайлам)."""
    rng = random.Random(seed)
    by_class: dict = defaultdict(list)
    for uid, info in intervals.items():
        by_class[info['class_name']].append(uid)

    uid_to_split: dict = {}
    for cls in classes_order:
        uids = sorted(by_class[cls])
        rng.shuffle(uids)
        n = len(uids)
        n_val  = max(1, round(n * val_frac))
        n_test = max(1, round(n * test_frac))
        for uid in uids[:n_val]:
            uid_to_split[uid] = 'val'
        for uid in uids[n_val:n_val + n_test]:
            uid_to_split[uid] = 'test'
        for uid in uids[n_val + n_test:]:
            uid_to_split[uid] = 'train'

    split_class_counts: dict = defaultdict(Counter)
    for uid, split in uid_to_split.items():
        split_class_counts[split][intervals[uid]['class_name']] += len(
            intervals[uid]['files'].get('ДС', [])
        )

    print('\nСплит (тайлов ДС по классам):')
    print(f'  {"Класс":<55} {"Train":>7} {"Val":>6} {"Test":>6}')
    print('  ' + '-' * 78)
    for cls in classes_order:
        tr = split_class_counts['train'].get(cls, 0)
        vl = split_class_counts['val'].get(cls, 0)
        te = split_class_counts['test'].get(cls, 0)
        print(f'  {cls:<55} {tr:>7,} {vl:>6,} {te:>6,}')
    tr_tot = sum(split_class_counts['train'].values())
    vl_tot = sum(split_class_counts['val'].values())
    te_tot = sum(split_class_counts['test'].values())
    tot = tr_tot + vl_tot + te_tot
    print(f'  {"ИТОГО":<55} {tr_tot:>7,} {vl_tot:>6,} {te_tot:>6,}')
    print(f'  Val={vl_tot/tot*100:.1f}%  Test={te_tot/tot*100:.1f}%')

    return uid_to_split


def split_well_based(intervals: dict, classes_order: list,
                     val_frac: float = 0.10, test_frac: float = 0.10,
                     min_class_tiles: int = 30) -> dict:
    """Well-based greedy split по интервалам (для raw-минералов — их не видно
    в структуре директорий, только через scan_tiles_by_interval, поэтому
    split_well_greedy() из БЛОКА 2а тут не подходит)."""
    wt: dict = defaultdict(Counter)
    for uid, info in intervals.items():
        n = len(info['files'].get('ДС', []))
        wt[info['well']][info['class_name']] += n

    wells = sorted(wt.keys())
    global_total: Counter = Counter()
    for w in wells:
        global_total.update(wt[w])

    unassigned = list(wells)
    split_counts: dict = {'train': Counter(), 'val': Counter(), 'test': Counter()}
    well_to_split: dict = {}

    def score(w, target_split, target_frac):
        trial = Counter(split_counts[target_split]) + Counter(wt[w])
        trial_frac = sum(trial.values()) / sum(global_total.values())
        dist = abs(trial_frac - target_frac)
        min_cls = min((trial.get(c, 0) for c in classes_order), default=0)
        return (dist, -min_cls)

    for target_split, target_frac in [('val', val_frac), ('test', test_frac)]:
        while unassigned:
            assigned = sum(split_counts[target_split].values())
            grand = sum(global_total.values())
            if (grand > 0 and assigned / grand >= target_frac and
                    all(split_counts[target_split].get(c, 0) >= min_class_tiles
                        for c in classes_order)):
                break
            best = min(unassigned, key=lambda w: score(w, target_split, target_frac))
            unassigned.remove(best)
            well_to_split[best] = target_split
            split_counts[target_split].update(wt[best])

    for w in unassigned:
        well_to_split[w] = 'train'
        split_counts['train'].update(wt[w])

    print('\nСплит по скважинам:')
    for split in ['train', 'val', 'test']:
        ws = sorted(w for w, s in well_to_split.items() if s == split)
        print(f'  {split}: {ws}')

    uid_to_split: dict = {}
    for uid, info in intervals.items():
        uid_to_split[uid] = well_to_split[info['well']]

    return uid_to_split


def build_dataset_from_intervals(intervals: dict, uid_to_split: dict,
                                 classes_order: list, out_dir: Path) -> None:
    if out_dir.exists():
        print(f'Удаляем {out_dir}')
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    total = 0
    for uid, info in tqdm(intervals.items(), desc='Копируем тайлы'):
        split = uid_to_split.get(uid)
        if split is None:
            continue
        cls = info['class_name']
        for mod, files in info['files'].items():
            dst_dir = out_dir / mod / split / cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            for src in files:
                shutil.copy2(src, dst_dir / src.name)
                total += 1

    print(f'Скопировано: {total:,} файлов → {out_dir}')

    class_to_idx = {cls: i for i, cls in enumerate(classes_order)}
    with open(out_dir / 'label_encoder.json', 'w', encoding='utf-8') as f:
        json.dump(class_to_idx, f, ensure_ascii=False, indent=2)
    print(f'Классов: {len(classes_order)}  →  {classes_order}')


def compute_normalization(out_dir: Path, n_samples: int = 3000, seed: int = 42) -> None:
    rng = random.Random(seed)
    stats: dict = {}

    for mod in MODALITIES:
        train_dir = out_dir / mod / 'train'
        if not train_dir.exists():
            continue
        all_files = []
        for cls_dir in train_dir.iterdir():
            if cls_dir.is_dir():
                all_files.extend(list(cls_dir.glob('*.jpg')))
        rng.shuffle(all_files)
        sample = all_files[:n_samples]

        pixels = []
        for p in tqdm(sample, desc=f'  Нормализация {mod}', leave=False):
            try:
                img = np.array(Image.open(p).convert('RGB')) / 255.0
                pixels.append(img.reshape(-1, 3))
            except Exception:
                pass

        if pixels:
            arr = np.concatenate(pixels, axis=0)
            mean = arr.mean(axis=0).tolist()
            std  = arr.std(axis=0).tolist()
        else:
            mean = [0.5, 0.5, 0.5]
            std  = [0.2, 0.2, 0.2]

        stats[mod] = {'mean': mean, 'std': std}
        print(f'  {mod}: mean={[round(v,4) for v in mean]}  std={[round(v,4) for v in std]}')

    with open(out_dir / 'normalization_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)


def build_variant_dataset(args) -> None:
    variant   = args.variant
    tile_root = Path(args.tile_root) if args.tile_root else _cfg.PIPELINE_TILES
    out_dir   = Path(args.out_dir) if args.out_dir else _cfg.DATA_ROOT / f'dataset_{variant.replace("-", "_")}'

    is_soft  = variant == 'soft'
    no_merge = variant.startswith('no-merge') or is_soft

    print(f'Вариант  : {variant}')
    print(f'Мин.тайлы: {args.min_tiles}')
    print(f'Выход    : {out_dir}')
    print(f'Тайлы    : {tile_root}')

    print('\n[1/6] Сканирование тайлов...')
    intervals = scan_tiles_by_interval(tile_root)
    print(f'  Найдено интервалов: {len(intervals):,}')
    print(f'  Уникальных скважин: {len({v["well"] for v in intervals.values()})}')

    print('\n[2/6] Назначение классов...')
    if is_soft:
        base_classes, soft_map = load_soft_labels_config()
        print(f'  База: {base_classes}')
        for uid, info in intervals.items():
            mineral = info['mineral']
            info['class_name'] = mineral if mineral in soft_map else None
    else:
        intervals = assign_classes(intervals, no_merge=no_merge)

    print('\n[3/6] Фильтрация редких классов...')
    min_t = args.min_tiles if (no_merge or is_soft) else 50
    intervals, classes_order = filter_classes(intervals, min_tiles=min_t)

    print('\n[4/6] Разбивка на train/val/test...')
    if variant == 'no-merge-well' or (args.split_mode == 'well'):
        uid_to_split = split_well_based(intervals, classes_order)
    else:
        uid_to_split = split_random(intervals, classes_order)

    print('\n[5/6] Копирование тайлов...')
    build_dataset_from_intervals(intervals, uid_to_split, classes_order, out_dir)

    print('\n[6/6] Вычисление нормализации...')
    compute_normalization(out_dir)

    if is_soft:
        _, soft_map = load_soft_labels_config()
        soft_subset = {cls: soft_map[cls] for cls in classes_order if cls in soft_map}
        with open(out_dir / 'soft_labels.json', 'w', encoding='utf-8') as f:
            json.dump(soft_subset, f, ensure_ascii=False, indent=2)
        with open(out_dir / 'base_classes.json', 'w', encoding='utf-8') as f:
            json.dump(load_soft_labels_config()[0], f, ensure_ascii=False, indent=2)
        print(f'\nsoft_labels.json → {len(soft_subset)} классов')

    print(f'\nГотово! Датасет: {out_dir}')
    if is_soft:
        print('Для обучения:')
        print(f'  python3 train.py run_soft_dual --mode dual --soft-labels --dataset-root {out_dir}')
    else:
        print('Для обучения:')
        print(f'  python3 train.py run_exp_{variant.replace("-", "_")} --dataset-root {out_dir}')


# ═════════════════════════════════════════════════════════════════════════════
# main
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='Единый пайплайн подготовки данных ML-Core-Kern-',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--variant', default='default',
                   choices=['default', 'random', 'no-merge', 'no-merge-well', 'soft'],
                   help='default: канонический (6 кл., сплит по скважинам, стадии 1-8); '
                        'random/no-merge/no-merge-well/soft: варианты поверх готовых тайлов')
    p.add_argument('--split-mode', default='random', choices=['random', 'well'],
                   help='Только для --variant soft: сплит по интервалам или по скважинам '
                        '(no-merge-well всегда well, остальные варианты имеют фиксированный сплит)')
    p.add_argument('--force', action='store_true',
                   help='Пересобрать с нуля (тайлы/датасет)')
    p.add_argument('--stages', default='1,2,3,4,5,6,7,8',
                   help='Только для --variant default: этапы через запятую')
    p.add_argument('--tile-cm', type=float, default=None)
    p.add_argument('--overlap-cm', type=float, default=None)
    p.add_argument('--min-tiles', type=int, default=None,
                   help='default: мин. тайлов класса в val/test (см. config.MIN_CLASS_TILES); '
                        'варианты: мин. тайлов класса всего, чтобы не быть отброшенным (default: 300)')
    p.add_argument('--sample-size', type=int, default=3000)
    p.add_argument('--no-visuals', action='store_true',
                   help='Только для --variant default: пропустить Stage 8')
    p.add_argument('--out-dir', default=None, help='Только для вариантов (не default)')
    p.add_argument('--tile-root', default=None, help='Только для вариантов (не default)')
    return p.parse_args()


def main():
    args = parse_args()
    if args.variant == 'default':
        stages = {int(s.strip()) for s in args.stages.split(',')}
        build_default_dataset(args, stages)
    else:
        if args.min_tiles is None:
            args.min_tiles = 300
        build_variant_dataset(args)


if __name__ == '__main__':
    main()
