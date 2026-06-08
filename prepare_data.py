"""
Пайплайн подготовки данных.

Запуск:
    python3 prepare_data.py [опции]

Примеры:
    python3 prepare_data.py --force              # пересобрать всё с нуля
    python3 prepare_data.py --stages 4,5,6,7     # только сплит + датасет + нормализация
    python3 prepare_data.py --tile-cm 5 --force  # другой размер тайла
    python3 prepare_data.py --min-tiles 50       # ослабить ограничение на сплит
"""

import argparse
import json
import re
import random
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


# ─────────────────────────────────────────────────────────────────────────────
# Маппинг классов (6 групп)
# Уголь → Аргиллит (только 82 тайла — мало для отдельного класса)
# ─────────────────────────────────────────────────────────────────────────────
CLASS_RULES: Dict[str, Optional[str]] = {
    # УДАЛЯЕМ
    'Алевролит_песчанистый':                                    None,
    'Песчаник_с_включениями_угля':                              None,
    'Известняк':                                                None,
    'Породы_фундамента':                                        None,

    # АРГИЛЛИТ (включая угольные разности)
    'Аргиллит':                                                 'Аргиллит',
    'Аргиллит_углистый':                                        'Аргиллит',
    'Аргиллит_с_включениями_угля':                              'Аргиллит',
    'Уголь':                                                    'Аргиллит',
    'Уголь_с_прослоями_аргиллита':                              'Аргиллит',

    # АЛЕВРОЛИТ
    'Алевролит':                                                'Алевролит',
    'Алевролит_с_включениями_угля':                             'Алевролит',
    'Алевролит_глинистый':                                      'Алевролит',
    'Алевролит_карбонатный':                                    'Алевролит',

    # КАРБОНАТ
    'Глинисто-карбонатная_порода':                              'Карбонат',
    'Опока_глинистая':                                          'Карбонат',
    'Глина_опоковидная':                                        'Карбонат',
    'Глина_аргиллитоподобная_с_прослоями_глины_опоковидной':    'Карбонат',
    'Кремнисто-глинистая_порода':                               'Карбонат',
    'Глина_опоковидная_с_включением_глинистых_опок':            'Карбонат',
    'Глина_аргиллитоподобная':                                  'Карбонат',

    # ПЕРЕСЛАИВАНИЕ С ПЕСЧАНИКОМ — светлое
    'Переслаивание_песчаника,_аргиллита_и_алевролита':          'Перес_светлое',
    'Песчаник_с_включениями_алевролита_и_аргиллита':            'Перес_светлое',
    'Чередование_аргиллита,_алевролита_и_песчаника':            'Перес_светлое',
    'Алевролит_с_прослоями_песчаника_и_аргиллита':              'Перес_светлое',
    'Песчаник_с_прослоями_алевролита_и_аргиллита':              'Перес_светлое',
    'Аргиллит_с_прослоями_песчаника_и_алевролита':              'Перес_светлое',
    'Песчаник_с_прослоями_алевролита':                          'Перес_светлое',
    'Переслаивание_песчаника_и_алевролита':                     'Перес_светлое',
    'Алевролит_с_прослоями_песчаника':                          'Перес_светлое',
    'Песчаник_с_прослоями_аргиллита':                           'Перес_светлое',
    'Аргиллит_с_прослоями_песчаника':                           'Перес_светлое',
    'Переслаивание_песчаника_и_аргиллита':                      'Перес_светлое',

    # ПЕРЕСЛАИВАНИЕ БЕЗ ПЕСЧАНИКА — тёмное
    'Переслаивание_аргиллита_и_алевролита':                     'Перес_тёмное',
    'Аргиллит_алевритовый':                                     'Перес_тёмное',
    'Алевролит_с_прослоями_аргиллита':                          'Перес_тёмное',
    'Аргиллит_с_прослоями_алевролита':                          'Перес_тёмное',

    # ПЕСЧАНИК (только чистый)
    'Песчаник':                                                 'Песчаник',
    'Песчаник_карбонатный':                                     'Песчаник',
}

IMG_EXTS   = {'.jpg', '.jpeg', '.png', '.bmp'}
TILE_MIN_H = 20
_DEPTH_RE  = re.compile(r'_([\d]+[.,][\d]+)\s*-\s*([\d]+[.,][\d]+)$')


# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции
# ─────────────────────────────────────────────────────────────────────────────

def _norm(name: str) -> str:
    return name.strip().replace(' ', '_')


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


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Excel → CSV
# ─────────────────────────────────────────────────────────────────────────────

def stage1(data_root: Path, csv_root: Path) -> None:
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


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Тайлинг
# ─────────────────────────────────────────────────────────────────────────────

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
            mineral_key = _norm(str(row['mineral']))
            if mineral_key not in CLASS_RULES:
                unknown_set.add(mineral_key)
                continue
            mapped_class = CLASS_RULES[mineral_key]
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
            step_px = max(1, int(round((tile_cm - overlap_cm) * ppc)))

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


def stage2(data_root: Path, csv_root: Path, tile_root: Path,
           modalities: List[str], tile_cm: float, overlap_cm: float,
           force: bool) -> None:
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
        print(f'  добавьте в CLASS_RULES: {sorted(all_unknown)}')


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Сканирование тайлов
# ─────────────────────────────────────────────────────────────────────────────

def stage3(tile_root: Path, classes_order: List[str]) -> Dict:
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


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — Стратифицированный сплит
# ─────────────────────────────────────────────────────────────────────────────

def _well_totals(wcc: Dict) -> Dict[str, Counter]:
    result = {}
    for well, mods in wcc.items():
        c: Counter = Counter()
        for cls_counts in mods.values():
            c.update(cls_counts)
        result[well] = c
    return result


def stage4(wcc: Dict, classes_order: List[str],
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

    # Реальный порог: min(min_class_tiles, global_count // 4) — не требовать больше, чем есть
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


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 — Копирование в датасет
# ─────────────────────────────────────────────────────────────────────────────

def stage5(tile_root: Path, ds_root: Path, well_to_split: Dict[str, str],
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


# ─────────────────────────────────────────────────────────────────────────────
# Stage 6 — Итоговая статистика
# ─────────────────────────────────────────────────────────────────────────────

def stage6(ds_root: Path, modalities: List[str], classes_order: List[str]) -> Dict:
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


# ─────────────────────────────────────────────────────────────────────────────
# Stage 7 — Нормализация
# ─────────────────────────────────────────────────────────────────────────────

def stage7(ds_root: Path, modalities: List[str], norm_stats_file: Path,
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


# ─────────────────────────────────────────────────────────────────────────────
# Stage 8 — Визуальные артефакты
# ─────────────────────────────────────────────────────────────────────────────

def stage8(ds_root: Path, classes_order: List[str], modalities: List[str],
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

    # class_distribution.png
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

    # tile_examples.png
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


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Пайплайн подготовки данных ML-Core-Kern-',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--force',       action='store_true',
                   help='Очистить tiles/ и dataset/ перед пересборкой')
    p.add_argument('--stages',      default='1,2,3,4,5,6,7,8',
                   help='Этапы через запятую (default: 1,2,3,4,5,6,7,8)')
    p.add_argument('--tile-cm',     type=float, default=None,
                   help=f'Высота тайла в см (default: {_cfg.TILE_CM} из config.py)')
    p.add_argument('--overlap-cm',  type=float, default=None,
                   help=f'Перекрытие в см (default: {_cfg.OVERLAP_CM} из config.py)')
    p.add_argument('--min-tiles',   type=int,   default=None,
                   help=f'Мин. тайлов класса в val/test (default: {_cfg.MIN_CLASS_TILES})')
    p.add_argument('--sample-size', type=int,   default=3000,
                   help='Сэмплов для вычисления нормализации (default: 3000)')
    p.add_argument('--no-visuals',  action='store_true',
                   help='Пропустить Stage 8 (визуальные артефакты)')
    return p.parse_args()


def main():
    args = parse_args()
    stages = {int(s.strip()) for s in args.stages.split(',')}

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

    print(f'CLASSES_ORDER : {classes_order}')
    print(f'tile_cm       : {tile_cm} см  |  overlap_cm: {overlap_cm} см')
    print(f'min_tiles     : {min_tiles}')
    print(f'Stages        : {sorted(stages)}')
    print()

    if 1 in stages:
        stage1(data_root, csv_root)

    if 2 in stages:
        stage2(data_root, csv_root, tile_root, modalities, tile_cm, overlap_cm, args.force)

    wcc = None
    if 3 in stages or 4 in stages:
        wcc = stage3(tile_root, classes_order)

    well_to_split = None
    if 4 in stages:
        well_to_split, _ = stage4(wcc, classes_order,
                                  _cfg.VAL_FRAC, _cfg.TEST_FRAC, _cfg.SEED, min_tiles)

    if 5 in stages:
        if well_to_split is None:
            raise RuntimeError('Stage 5 требует Stage 4 (добавьте 4 в --stages)')
        stage5(tile_root, ds_root, well_to_split, modalities, classes_order, args.force)

    if 6 in stages:
        stage6(ds_root, modalities, classes_order)

    if 7 in stages:
        stage7(ds_root, modalities, norm_stats, args.sample_size, _cfg.SEED)

    if 8 in stages and not args.no_visuals:
        stage8(ds_root, classes_order, modalities, dataset_stats, tile_cm, _cfg.SEED)

    print('\nГотово.')


if __name__ == '__main__':
    main()
