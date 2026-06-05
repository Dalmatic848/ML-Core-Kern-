"""
Диагностический пайплайн данных — использует уже готовые тайлы из data/pipeline/tiles/.

Режимы:
  random    — стандартные 6 классов, но случайный (интервальный) сплит 80/10/10
  no-merge  — оригинальные минералы как классы (без слияния), случайный сплит
  no-merge-well — оригинальные минералы, well-based сплит

Примеры:
  python3 prepare_exp.py --mode random
  python3 prepare_exp.py --mode no-merge --min-tiles 300
  python3 prepare_exp.py --mode random --out-dir data/dataset_random
"""

import argparse
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

# ──────────────────────────────────────────────────────────────────────────────
# Стандартный 6-классовый маппинг (из prepare_data.py)
# ──────────────────────────────────────────────────────────────────────────────
MINERAL_TO_6CLASS = {
    'Аргиллит': 'Аргиллит',
    'Аргиллит_углистый': 'Аргиллит',
    'Аргиллит_с_включениями_угля': 'Аргиллит',
    'Уголь': 'Аргиллит',
    'Уголь_с_прослоями_аргиллита': 'Аргиллит',
    'Алевролит': 'Алевролит',
    'Алевролит_с_включениями_угля': 'Алевролит',
    'Алевролит_глинистый': 'Алевролит',
    'Алевролит_карбонатный': 'Алевролит',
    'Глинисто-карбонатная_порода': 'Карбонат',
    'Опока_глинистая': 'Карбонат',
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
    'Песчаник': 'Песчаник',
    'Песчаник_карбонатный': 'Песчаник',
}

FRAG_RE = re.compile(r'_(\d+\.\d{3})_(\d+\.\d{3})_frag\d{4}\.jpg$')
MODALITIES = ['ДС', 'УФ']
SEED = cfg.SEED

# ──────────────────────────────────────────────────────────────────────────────
# Мягкие метки: каждый минерал → вектор [Пес, Алев, Арг, Карб, Гл_оп]
# Логика: "с_прослоями X" = 75% основного + 25% X;
#         "с_прослоями X_и_Y" = 60%+20%+20%; Переслаивание = равные доли.
# ──────────────────────────────────────────────────────────────────────────────
BASE_CLASSES = ['Песчаник', 'Алевролит', 'Аргиллит', 'Карбонат', 'Глина_опоковидная']

SOFT_LABEL_MAP: Dict[str, Optional[list]] = {
    # ── Песчаник ──────────────────────────────────────────────────────────────
    'Песчаник':                                         [1.00, 0.00, 0.00, 0.00, 0.00],
    'Песчаник_карбонатный':                             [0.70, 0.00, 0.00, 0.30, 0.00],
    'Песчаник_с_прослоями_алевролита':                  [0.75, 0.25, 0.00, 0.00, 0.00],
    'Песчаник_с_прослоями_аргиллита':                   [0.75, 0.00, 0.25, 0.00, 0.00],
    'Песчаник_с_прослоями_алевролита_и_аргиллита':      [0.60, 0.20, 0.20, 0.00, 0.00],
    'Песчаник_с_включениями_алевролита_и_аргиллита':    [0.60, 0.20, 0.20, 0.00, 0.00],
    # ── Алевролит ─────────────────────────────────────────────────────────────
    'Алевролит':                                        [0.00, 1.00, 0.00, 0.00, 0.00],
    'Алевролит_карбонатный':                            [0.00, 0.70, 0.00, 0.30, 0.00],
    'Алевролит_глинистый':                              [0.00, 0.80, 0.20, 0.00, 0.00],
    'Алевролит_с_включениями_угля':                     [0.00, 0.90, 0.10, 0.00, 0.00],
    'Алевролит_с_прослоями_песчаника':                  [0.25, 0.75, 0.00, 0.00, 0.00],
    'Алевролит_с_прослоями_аргиллита':                  [0.00, 0.75, 0.25, 0.00, 0.00],
    'Алевролит_с_прослоями_песчаника_и_аргиллита':      [0.20, 0.60, 0.20, 0.00, 0.00],
    # ── Аргиллит ──────────────────────────────────────────────────────────────
    'Аргиллит':                                         [0.00, 0.00, 1.00, 0.00, 0.00],
    'Аргиллит_углистый':                                [0.00, 0.00, 1.00, 0.00, 0.00],
    'Аргиллит_с_включениями_угля':                      [0.00, 0.00, 1.00, 0.00, 0.00],
    'Аргиллит_алевритовый':                             [0.00, 0.30, 0.70, 0.00, 0.00],
    'Аргиллит_с_прослоями_алевролита':                  [0.00, 0.25, 0.75, 0.00, 0.00],
    'Аргиллит_с_прослоями_песчаника':                   [0.25, 0.00, 0.75, 0.00, 0.00],
    'Аргиллит_с_прослоями_песчаника_и_алевролита':      [0.20, 0.20, 0.60, 0.00, 0.00],
    'Глина_аргиллитоподобная':                          [0.00, 0.00, 1.00, 0.00, 0.00],
    'Уголь':                                            [0.00, 0.00, 1.00, 0.00, 0.00],
    'Уголь_с_прослоями_аргиллита':                      [0.00, 0.00, 1.00, 0.00, 0.00],
    # ── Переслаивания ─────────────────────────────────────────────────────────
    'Переслаивание_аргиллита_и_алевролита':             [0.00, 0.50, 0.50, 0.00, 0.00],
    'Переслаивание_песчаника_и_аргиллита':              [0.50, 0.00, 0.50, 0.00, 0.00],
    'Переслаивание_песчаника_и_алевролита':             [0.50, 0.50, 0.00, 0.00, 0.00],
    'Переслаивание_песчаника,_аргиллита_и_алевролита':  [0.34, 0.33, 0.33, 0.00, 0.00],
    'Чередование_аргиллита,_алевролита_и_песчаника':    [0.34, 0.33, 0.33, 0.00, 0.00],
    # ── Карбонат ──────────────────────────────────────────────────────────────
    'Глинисто-карбонатная_порода':                      [0.00, 0.00, 0.10, 0.90, 0.00],
    'Глина_аргиллитоподобная_с_прослоями_глины_опоковидной': [0.00, 0.00, 0.50, 0.00, 0.50],
    'Кремнисто-глинистая_порода':                       [0.00, 0.00, 0.30, 0.00, 0.70],
    # ── Глина опоковидная ─────────────────────────────────────────────────────
    'Глина_опоковидная':                                [0.00, 0.00, 0.00, 0.00, 1.00],
    'Глина_опоковидная_с_включением_глинистых_опок':    [0.00, 0.00, 0.00, 0.00, 1.00],
    'Опока_глинистая':                                  [0.00, 0.00, 0.00, 0.00, 1.00],
}


# ──────────────────────────────────────────────────────────────────────────────
# Шаг 1 — Сканирование существующих тайлов по именам файлов
# ──────────────────────────────────────────────────────────────────────────────
def scan_tiles(tile_root: Path) -> dict:
    """
    Возвращает dict: interval_uid -> {well, mineral, files: {mod: [path, ...]}}
    Каждый interval_uid = '{well}||{mineral}_{d_from}_{d_to}'
    """
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


# ──────────────────────────────────────────────────────────────────────────────
# Шаг 2 — Назначение класса каждому интервалу
# ──────────────────────────────────────────────────────────────────────────────
def assign_classes(intervals: dict, no_merge: bool) -> dict:
    """
    Добавляет 'class_name' к каждому интервалу.
    no_merge=True: используем исходный mineral name
    no_merge=False: маппим через MINERAL_TO_6CLASS
    """
    skipped = 0
    for uid, info in intervals.items():
        mineral = info['mineral']
        if no_merge:
            info['class_name'] = mineral
        else:
            mapped = MINERAL_TO_6CLASS.get(mineral)
            if mapped is None:
                info['class_name'] = None  # будет отфильтровано
                skipped += 1
            else:
                info['class_name'] = mapped
    if skipped:
        print(f'  [skip] {skipped} интервалов с неизвестным/удалённым минералом')
    return intervals


# ──────────────────────────────────────────────────────────────────────────────
# Шаг 3 — Фильтрация редких классов
# ──────────────────────────────────────────────────────────────────────────────
def filter_classes(intervals: dict, min_tiles: int) -> tuple:
    # Считаем тайлы по классам (берём только ДС для подсчёта)
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

    # Фильтруем
    filtered = {uid: info for uid, info in intervals.items()
                if info['class_name'] in kept_classes}
    return filtered, sorted(kept_classes)


# ──────────────────────────────────────────────────────────────────────────────
# Шаг 4 — Сплит
# ──────────────────────────────────────────────────────────────────────────────
def split_random(intervals: dict, classes_order: list,
                 val_frac: float = 0.10, test_frac: float = 0.10,
                 seed: int = SEED) -> dict:
    """Стратифицированный случайный сплит по интервалам (не по тайлам)."""
    rng = random.Random(seed)

    # Группируем uid по классу
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

    # Статистика
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
    """Well-based greedy split (упрощённая версия из prepare_data.py)."""
    # Собираем тайлы ДС per well per class
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


# ──────────────────────────────────────────────────────────────────────────────
# Шаг 5 — Копирование в датасет
# ──────────────────────────────────────────────────────────────────────────────
def build_dataset(intervals: dict, uid_to_split: dict,
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

    # label_encoder.json
    class_to_idx = {cls: i for i, cls in enumerate(classes_order)}
    with open(out_dir / 'label_encoder.json', 'w', encoding='utf-8') as f:
        json.dump(class_to_idx, f, ensure_ascii=False, indent=2)
    print(f'Классов: {len(classes_order)}  →  {classes_order}')


# ──────────────────────────────────────────────────────────────────────────────
# Шаг 6 — Нормализация
# ──────────────────────────────────────────────────────────────────────────────
def compute_normalization(out_dir: Path, n_samples: int = 3000, seed: int = SEED) -> None:
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


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', default='random',
                   choices=['random', 'no-merge', 'no-merge-well', 'soft'],
                   help='random: 6 кл. случайный сплит; no-merge: исходные минералы; '
                        'soft: мягкие метки (5 базовых компонент)')
    p.add_argument('--min-tiles', type=int, default=300,
                   help='Мин. тайлов ДС для класса (default: 300)')
    p.add_argument('--out-dir', default=None,
                   help='Папка датасета (по умолчанию: data/dataset_{mode})')
    p.add_argument('--tile-root', default='data/pipeline/tiles',
                   help='Папка с тайлами (default: data/pipeline/tiles)')
    args = p.parse_args()

    random.seed(SEED)
    tile_root = Path(args.tile_root)
    out_dir = Path(args.out_dir) if args.out_dir else Path(f'data/dataset_{args.mode.replace("-", "_")}')

    is_soft   = args.mode == 'soft'
    no_merge  = args.mode.startswith('no-merge') or is_soft

    print(f'Режим    : {args.mode}')
    print(f'Мин.тайлы: {args.min_tiles}')
    print(f'Выход    : {out_dir}')
    print(f'Тайлы    : {tile_root}')
    if is_soft:
        print(f'База     : {BASE_CLASSES}')

    # 1. Сканируем
    print('\n[1/6] Сканирование тайлов...')
    intervals = scan_tiles(tile_root)
    print(f'  Найдено интервалов: {len(intervals):,}')
    print(f'  Уникальных скважин: {len({v["well"] for v in intervals.values()})}')

    # 2. Назначаем классы
    print('\n[2/6] Назначение классов...')
    if is_soft:
        # В soft режиме оставляем оригинальные минералы, фильтруем по SOFT_LABEL_MAP
        for uid, info in intervals.items():
            mineral = info['mineral']
            info['class_name'] = mineral if mineral in SOFT_LABEL_MAP else None
    else:
        intervals = assign_classes(intervals, no_merge=no_merge)

    # 3. Фильтруем редкие
    print('\n[3/6] Фильтрация редких классов...')
    min_t = args.min_tiles if (no_merge or is_soft) else 50
    intervals, classes_order = filter_classes(intervals, min_tiles=min_t)

    # 4. Сплит
    print('\n[4/6] Разбивка на train/val/test...')
    if args.mode == 'no-merge-well':
        uid_to_split = split_well_based(intervals, classes_order)
    else:
        uid_to_split = split_random(intervals, classes_order)

    # 5. Строим датасет
    print('\n[5/6] Копирование тайлов...')
    build_dataset(intervals, uid_to_split, classes_order, out_dir)

    # 6. Нормализация
    print('\n[6/6] Вычисление нормализации...')
    compute_normalization(out_dir)

    # 7. Для soft режима — сохраняем soft_labels.json и base_classes.json
    if is_soft:
        soft_subset = {cls: SOFT_LABEL_MAP[cls] for cls in classes_order if cls in SOFT_LABEL_MAP}
        with open(out_dir / 'soft_labels.json', 'w', encoding='utf-8') as f:
            json.dump(soft_subset, f, ensure_ascii=False, indent=2)
        with open(out_dir / 'base_classes.json', 'w', encoding='utf-8') as f:
            json.dump(BASE_CLASSES, f, ensure_ascii=False, indent=2)
        print(f'\nsoft_labels.json → {len(soft_subset)} классов')
        print(f'base_classes.json → {BASE_CLASSES}')

    print(f'\nГотово! Датасет: {out_dir}')
    if is_soft:
        print(f'Для обучения:')
        print(f'  python3 train.py run_soft_dual --mode dual --soft-labels --dataset-root {out_dir}')
    else:
        print(f'Для обучения:')
        print(f'  python3 train.py run_exp_{args.mode.replace("-", "_")} --dataset-root {out_dir}')


if __name__ == '__main__':
    main()
