"""
Центральный конфиг проекта ML-Core-Kern-.
Все пути, гиперпараметры и визуальные константы берутся отсюда.
"""

import json as _json
from pathlib import Path

# ── Корень проекта ────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent

# ── Пути к данным ─────────────────────────────────────────────────────────────
DATA_ROOT       = ROOT / "data"
DIGITAL_CORE    = DATA_ROOT / "Digital_core"
PIPELINE_CSV    = DATA_ROOT / "pipeline" / "csv"
PIPELINE_TILES  = DATA_ROOT / "pipeline" / "tiles"
DATASET_ROOT    = DATA_ROOT / "dataset"
LABEL_ENCODER   = DATASET_ROOT / "label_encoder.json"
SPLIT_MANIFEST  = DATASET_ROOT / "split_manifest.json"
NORM_STATS_FILE = DATASET_ROOT / "normalization_stats.json"

# ── Пути к результатам ────────────────────────────────────────────────────────
RESULTS_ROOT    = ROOT / "results"
MODELS_ROOT     = ROOT / "models"
DATASET_STATS   = RESULTS_ROOT / "dataset_stats"

# ── Параметры тайлинга ────────────────────────────────────────────────────────
TILE_CM     = 10  # высота тайла в сантиметрах (10 см — виден прослой)
OVERLAP_CM  = 0    # нет перекрытия — убираем квазидубликаты

# ── Параметры датасета ────────────────────────────────────────────────────────
VAL_FRAC         = 0.10
TEST_FRAC        = 0.10
SEED             = 42
MIN_CLASS_TILES  = 100   # минимум тайлов каждого класса в val и test

# ── Таксономия (6 классов) ────────────────────────────────────────────────────
# Канонический источник — src/taxonomy.py (маппинг минерал→класс живёт там,
# импортируется в prepare_dataset.py и visualize_soft.py вместо трёх копий).
from src.taxonomy import CLASSES_ORDER  # noqa: E402,F401 (реэкспорт для config.CLASSES_ORDER)

# ── Параметры обучения ────────────────────────────────────────────────────────
BATCH_SIZE      = 64
MAX_EPOCHS      = 60
PATIENCE        = 15
MIN_DELTA       = 0.001
NUM_WORKERS     = 4
IMG_SIZE        = 224
WARMUP_EPOCHS   = 15   # pct_start для OneCycleLR = 15/60 = 0.25
DUAL_BATCH_SIZE = 32   # два ResNet18 backbone в памяти
DUAL_LR         = 3e-4

# ── ResNet50 (больше памяти → меньше батч) ────────────────────────────────────
RN50_BATCH_SIZE      = 32   # single ResNet50 (2048 features vs 512 у RN18)
RN50_DUAL_BATCH_SIZE = 16   # два ResNet50 backbone в памяти (4096 на fusion, RTX 3050)
RN50_LR              = 3e-4

# ── Нормализация: загружаем из файла если есть, иначе ImageNet ────────────────
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

# Публичные алиасы для обратной совместимости с visual_1m.ipynb
IMAGENET_MEAN = _IMAGENET_MEAN
IMAGENET_STD  = _IMAGENET_STD

if NORM_STATS_FILE.exists():
    with open(NORM_STATS_FILE, encoding='utf-8') as _f:
        _ns = _json.load(_f)
    DS_MEAN = _ns.get('ДС', {}).get('mean', _IMAGENET_MEAN)
    DS_STD  = _ns.get('ДС', {}).get('std',  _IMAGENET_STD)
    UV_MEAN = _ns.get('УФ', {}).get('mean', _IMAGENET_MEAN)
    UV_STD  = _ns.get('УФ', {}).get('std',  _IMAGENET_STD)
else:
    DS_MEAN = UV_MEAN = _IMAGENET_MEAN
    DS_STD  = UV_STD  = _IMAGENET_STD

# ── Конфиги моделей по модальности ───────────────────────────────────────────
MODEL_CONFIGS = {
    "ДС": dict(lr=3e-4, wd=1e-4, dropout=0.5, freeze="none", resize="square", aug="heavy",
               loss_type="focal", use_sampler=False, mix_aug=True, clip_grad=1.0,
               scheduler="onecycle", focal_gamma=2.0),
    "УФ": dict(lr=3e-4, wd=1e-4, dropout=0.5, freeze="none", resize="square", aug="std",
               loss_type="focal", use_sampler=False, mix_aug=True, clip_grad=1.0,
               scheduler="onecycle", focal_gamma=2.0),
}
MODALITIES = list(MODEL_CONFIGS.keys())

# ── Regex для парсинга глубин из имени файла ──────────────────────────────────
DEPTH_RE = r"(\d+[.,]\d+)\s*-\s*(\d+[.,]\d+)"

# ── Цветовая палитра классов ──────────────────────────────────────────────────
CLASS_PALETTE = {
    "Песчаник":       "#3B82F6",
    "Аргиллит":       "#EF4444",
    "Алевролит":      "#F59E0B",
    "Перес_светлое":  "#8B5CF6",
    "Перес_тёмное":   "#4C1D95",
    "Карбонат":       "#292524",
    # backward compat
    "Уголь":          "#1C1917",
    "Переслаивание":  "#8B5CF6",
    "unknown":        "#E5E7EB",
}

# ── Короткие названия классов для графиков ────────────────────────────────────
CLASS_SHORT = {
    "Песчаник":       "Пс",
    "Аргиллит":       "Ар",
    "Алевролит":      "Ал",
    "Перес_светлое":  "Пер_П",
    "Перес_тёмное":   "Пер_А",
    "Карбонат":       "Карб.",
    # backward compat
    "Уголь":          "Уг",
    "Переслаивание":  "Перес.",
}

# ── Схема сохранения результатов ──────────────────────────────────────────────
# results/
# └── {run_name}/
#     ├── config.json
#     ├── ДС_best.pth / УФ_best.pth
#     ├── ДС_history.json / УФ_history.json
#     ├── ДС_test_metrics.json / УФ_test_metrics.json
#     └── plots/
#         ├── curves_{mod}.png
#         ├── confusion_{mod}.png
#         ├── per_class_f1_{mod}.png
#         └── metrics_summary.png
