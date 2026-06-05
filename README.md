# ML-Core-Kern — Литотипизация керна

Классификация литотипов горных пород по фотографиям бурового керна.  
Две модальности: **ДС** (дневной свет) и **УФ** (ультрафиолет).

---

## Задача

По фотографии фрагмента керна (тайл 10 см) определить литотип породы.  
**6 классов**:

| # | Класс | Описание |
|---|-------|---------|
| 0 | Песчаник | Только чистый песчаник |
| 1 | Аргиллит | Аргиллит + угольные разности (уголь — только 82 тайла, мало для отдельного класса) |
| 2 | Алевролит | Чистый алевролит и разновидности |
| 3 | Перес_светлое | Переслаивание с прослоями песчаника (светлые текстуры) |
| 4 | Перес_тёмное | Переслаивание аргиллит+алевролит без песчаника (тёмные текстуры) |
| 5 | Карбонат | Глинисто-карбонатные и кремнисто-глинистые породы |

Обучаются три варианта моделей:
- **ДС** — ResNet18 / ResNet50, только дневной свет
- **УФ** — ResNet18 / ResNet50, только ультрафиолет
- **Dual** — DualStreamResNet18 / DualStreamResNet50, совместно ДС + УФ

---

## Таксономия: принцип объединения

**Из исходных геологических разностей → 6 классов** по принципу визуальной отличимости на тайле.

| Было | Стало | Причина |
|------|-------|---------|
| Уголь, Аргиллит_углистый, Аргиллит_с_включениями_угля, Уголь_с_прослоями_аргиллита | → Аргиллит | Всего 82 тайла Угля — недостаточно для самостоятельного класса |
| Переслаивание_аргиллита_и_алевролита, Аргиллит_алевритовый, ... (4 разности) | → Перес_тёмное | Тёмная слоистость без песчаника |
| Переслаивание_п+а+ал, Песчаник_с_прослоями_*, Чередование_*, ... (12 разностей) | → Перес_светлое | Светлая слоистость с прослоями песчаника |
| Глинисто-карбонатная_порода + 6 вариантов | → Карбонат | Единый визуальный тип |
| Песчаник + Песчаник_карбонатный | → Песчаник | Визуально идентичны |

Разделение единого **Переслаивания** на два подкласса — ключевое улучшение v4:  
старое слияние создавало гетерогенный класс с ~49% тайлов, что смещало все модели к нему.

---

## Структура проекта

```
ML-Core-Kern-/
├── config.py                    # Единый конфиг: пути, гиперпараметры, цвета, CLASSES_ORDER
├── prepare_data.py              # CLI: подготовка данных (все 8 стадий)
├── train.py                     # CLI: ЕДИНЫЙ скрипт обучения (все арх. и режимы)
├── visualize_results.py         # CLI: визуализация на фото + метрики
│
│   # Устаревшие скрипты обучения (заменены train.py, оставлены для совместимости):
├── train_run.py                 #   ResNet18 single
├── train_multimodal.py          #   ResNet18 dual
├── train_run_rn50.py            #   ResNet50 single
├── train_multimodal_rn50.py     #   ResNet50 dual
├── data/
│   ├── Digital_core/            # Исходные фото + Excel (raw)
│   ├── pipeline/
│   │   ├── csv/                 # Excel → CSV (Stage 1)
│   │   └── tiles/               # Тайлы по скважинам (Stage 2)
│   └── dataset/                 # Финальный датасет
│       ├── ДС/{train,val,test}/{class}/
│       ├── УФ/{train,val,test}/{class}/
│       ├── label_encoder.json
│       ├── split_manifest.json
│       └── normalization_stats.json
├── src/
│   ├── utils.py
│   ├── transforms.py            # get_transforms (resize, aug, mean/std)
│   ├── data.py                  # prepare_loaders, PairedDataset, prepare_paired_loaders
│   ├── models/
│   │   ├── resnet.py            # create_resnet18, create_resnet50
│   │   └── dual_resnet.py       # DualStreamResNet18/50, create_dual_resnet18/50
│   ├── training.py              # train_one_epoch (+ batch_scheduler), validate, EarlyStopping
│   ├── losses.py                # FocalLoss (+ weight, gamma), LabelSmoothingCE, get_criterion
│   └── augmentation.py          # cutmix_data, cutmix_data_paired
├── notebooks/                   # ⚠️ УСТАРЕВШИЕ АРТЕФАКТЫ — не использовать
│   ├── preparation.ipynb        #    (заменён prepare_data.py)
│   └── visual_1m.ipynb          #    (заменён visualize_results.py)
└── results/
    ├── run_v3/                  # Single-stream, 5 классов (лучший single-stream)
    ├── run_v4/                  # Dual-stream, 6 классов (текущий)
    └── dataset_stats/
```

---

## Пайплайн подготовки данных (`prepare_data.py`)

```bash
python3 prepare_data.py [--force] [--stages 1,2,3,4,5,6,7,8] [--tile-cm N] [--min-tiles N]

# Примеры:
python3 prepare_data.py --force                    # пересборка с нуля
python3 prepare_data.py --stages 4,5,6,7           # только сплит + датасет + нормализация
python3 prepare_data.py --tile-cm 5 --force        # другой размер тайла
python3 prepare_data.py --min-tiles 50             # ослабить ограничение сплита
```

8 стадий:

| Stage | Описание |
|-------|---------|
| 1 | Excel → CSV (парсинг колонок B:G, skiprows=8) |
| 2 | Кроп + склейка + тайлинг (TILE_CM=10, OVERLAP_CM=0) |
| 3 | Сканирование тайлов → матрица well × class × modality |
| 4 | Стратифицированный жадный сплит по скважинам |
| 5 | Копирование тайлов в dataset/{modality}/{split}/{class}/ |
| 6 | Итоговая статистика |
| 7 | Нормализация: mean/std из 3000 train-тайлов → normalization_stats.json |
| 8 | Визуальные артефакты (class_distribution.png, tile_examples.png) |

**Актуальный размер датасета (10 см, без перекрытия):**

| Сплит | ДС | УФ |
|-------|----|----|
| train | ~13 000 | ~13 000 |
| val | ~2 100 | ~2 100 |
| test | ~2 800 | ~2 800 |

Дисбаланс train: ~5:1 (Перес_светлое vs Алевролит).

---

## Архитектуры

### Single-stream: ResNet18 / ResNet50

```
ResNet18: fc = Dropout → Linear(512 → num_classes)
ResNet50: fc = Dropout → Linear(2048 → num_classes)
```

Две независимые модели — по одной на ДС и УФ.

### Dual-stream: DualStreamResNet18 / DualStreamResNet50

```
ResNet18: backbone_ds/uv → (B, 512)   cat → (B, 1024) → Linear(1024→512) → ReLU → Dropout → Linear(512→N)
ResNet50: backbone_ds/uv → (B, 2048)  cat → (B, 4096) → Linear(4096→1024) → ReLU → Dropout → Linear(1024→N)
```

Оба потока обучаются совместно end-to-end. Синхронный CutMix: одна bbox-маска применяется к обоим потокам одновременно.

---

## Обучение

### `train.py` — единый скрипт (основной)

```bash
python3 train.py run_v5                                              # ResNet18, ДС+УФ
python3 train.py run_v5         --mode dual                         # ResNet18, dual
python3 train.py run_rn50_v1    --arch resnet50                     # ResNet50, ДС+УФ
python3 train.py run_rn50_dual  --arch resnet50 --mode dual         # ResNet50, dual
python3 train.py run_warm       --arch resnet50 --mode dual \
                                --warm-start results/run_rn50_v1    # тёплый старт
```

**Ручная остановка**: введите `q` + Enter во время обучения — завершение после текущей эпохи.

### Конфигурация (`config.py`)

```python
MODEL_CONFIGS = {
    'ДС': dict(lr=3e-4, wd=1e-4, dropout=0.5, aug='heavy',
               loss_type='focal', focal_gamma=2.0, scheduler='onecycle',
               use_sampler=False, mix_aug=True, clip_grad=1.0),
    'УФ': dict(lr=3e-4, wd=1e-4, dropout=0.5, aug='std',
               loss_type='focal', focal_gamma=2.0, scheduler='onecycle',
               use_sampler=False, mix_aug=True, clip_grad=1.0),
}
MAX_EPOCHS=60 | PATIENCE=15 | BATCH_SIZE=64 | DUAL_BATCH_SIZE=32 | WARMUP_EPOCHS=15
# ResNet50:
RN50_BATCH_SIZE=32 | RN50_DUAL_BATCH_SIZE=16 | RN50_LR=3e-4
```

| Компонент | Значение | Причина |
|-----------|---------|---------|
| Оптимизатор | Adam, lr=3e-4 | OneCycleLR стартует с lr/10 = 3e-5 |
| LR-расписание | OneCycleLR (pct_start=0.25) | Warmup 15 эп., затем cosine decay |
| Loss | FocalLoss (γ=2, class_weights) | Боремся с дисбалансом классов |
| Аугментация | CutMix + ColorJitter + HFlip + Affine | Снижение переобучения |
| Нормализация | из normalization_stats.json | DS mean≈0.58, UV mean≈0.19 |

---

## Визуализация (`visualize_results.py`)

```bash
# Статистика на test-сете (confusion matrix, F1, classification report)
python3 visualize_results.py run_v4 --dual --no-photos

# Визуализация конкретной скважины и глубины
python3 visualize_results.py run_v4 --dual --well Харасавэйск_2000 --depth 1613.00

# Только первые 2 скважины из тест-сета
python3 visualize_results.py run_v4 --dual --wells 2

# Single-stream модели, val-сет
python3 visualize_results.py run_v3 --split val

# Старые веса (5 классов) — num_classes определяется автоматически из checkpoint
python3 visualize_results.py run_v3
```

Сохраняет в `results/{run}/visual/stats/`:
- `confusion_test.png` — матрица ошибок
- `per_class_f1_test.png` — F1 по классам
- `confidence_distribution.png` — уверенность: верные vs ошибочные
- `classification_report.png` — precision / recall / F1 / support

---

## Сводная таблица экспериментов

| Run | Архитектура | Классов | Val F1 (лучшая) | Эпоха | Статус |
|-----|-------------|---------|-----------------|-------|--------|
| run_v3 ДС | ResNet18 single | 5 | **0.488** | 6 | baseline, well-based split |
| run_v3 УФ | ResNet18 single | 5 | **0.418** | 7 | baseline, well-based split |
| run_v4 Dual | ResNet18 dual | 6 | 0.452 | 2 | ненадёжно (2 эп.) |
| run_rn50_v1 ДС | ResNet50 single | 6 | **0.481** | 19 | полное обучение |
| run_rn50_v1 УФ | ResNet50 single | 6 | 0.377 | 2 | ⚠ прервано (4 эп.) |
| run_rn50_dual Dual | ResNet50 dual | 6 | **0.478** | 5 | ⚠ прервано (7 эп.) |
| run_random_dual Dual | ResNet18 dual | 6 | **0.569** | 2 | ⚠ прервано (4 эп.), random split |
| run_no_merge_dual Dual | ResNet18 dual | 18 | 0.242 | 5 | ⚠ прервано (6 эп.), random split |

---

## Текущие результаты

### run_v3 — ResNet18 Single, 5 классов (baseline)

| Модальность | F1 macro (val) | Acc | Best epoch |
|-------------|----------------|-----|------------|
| ДС | **0.488** | 0.712 | 6 |
| УФ | **0.418** | 0.643 | 7 |

Проблемы: Алевролит F1≈0, best epoch 6–7 из 60 (ReduceLROnPlateau останавливал слишком рано).

---

### run_v4 — ResNet18 Dual, 6 классов

| Модель | F1 macro (val) | Acc | Best epoch |
|--------|----------------|-----|------------|
| Dual ДС+УФ | 0.452 | 0.610 | **2** |

Результаты ненадёжны: датасет перестраивался в процессе (7→6 классов), best epoch = 2. Нужен чистый запуск.

---

### run_rn50_v1 — ResNet50 Single, 6 классов

| Модальность | F1 macro (val) | Acc (val) | F1 macro (test) | Acc (test) | Best epoch |
|-------------|----------------|-----------|-----------------|------------|------------|
| ДС | **0.481** | 0.623 | 0.260 | 0.279 | 19 |
| УФ | 0.377 | 0.518 | 0.190 | 0.157 | 2 ⚠ |

⚠ **УФ прервана после 4 эпох** — результат неполный, требует дообучения.

**F1 по классам (test, ДС):**

| Класс | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Песчаник | 0.67 | 0.49 | **0.57** |
| Аргиллит | 0.31 | 0.38 | 0.34 |
| Карбонат | 0.12 | 0.67 | 0.21 |
| Перес_тёмное | 0.15 | 0.46 | 0.22 |
| Перес_светлое | 0.60 | 0.10 | 0.18 |
| Алевролит | 0.02 | 0.09 | 0.04 |

Val F1=0.481 сопоставим с ResNet18 (0.488) — более глубокая архитектура не даёт преимущества при неполном обучении УФ.

---

### run_rn50_dual — ResNet50 Dual, 6 классов (тёплый старт из run_rn50_v1)

| Модель | F1 macro (val) | Acc (val) | F1 macro (test) | Acc (test) | Best epoch |
|--------|----------------|-----------|-----------------|------------|------------|
| Dual ДС+УФ | **0.478** | 0.606 | 0.285 | 0.280 | 5 ⚠ |

⚠ **Прервано после 7 эпох** — результат неполный.

**F1 по классам (test):**

| Класс | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Песчаник | 0.69 | 0.50 | **0.58** |
| Аргиллит | 0.42 | 0.54 | **0.47** |
| Перес_тёмное | 0.16 | 0.33 | 0.22 |
| Карбонат | 0.14 | 0.33 | 0.19 |
| Перес_светлое | 0.67 | 0.10 | 0.18 |
| Алевролит | 0.04 | 0.30 | 0.07 |

Dual слегка улучшает Аргиллит (0.47 vs 0.34 у ДС single) — УФ добавляет диагностическую информацию.

---

### run_random_dual — ResNet18 Dual, 6 классов, случайный сплит

Диагностический эксперимент: те же 6 классов, та же архитектура, но сплит **по интервалам** (80/10/10), а не по скважинам. Датасет: `data/dataset_random`.

| Модель | Val F1 | Val Acc | Test F1¹ | Test Acc | Best epoch |
|--------|--------|---------|----------|----------|------------|
| Dual ДС+УФ | **0.569** | 0.638 | **0.531** | 0.579 | 2 ⚠ |

¹ Test из того же random split — скважины частично видела модель во время обучения.

⚠ **Прервано после 4 эпох** — модель не дообучена, OneCycleLR ещё в фазе warmup.

**F1 по классам (random test):**

| Класс | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Карбонат | 0.78 | 0.90 | **0.83** |
| Песчаник | 0.89 | 0.66 | **0.76** |
| Перес_светлое | 0.90 | 0.52 | 0.66 |
| Аргиллит | 0.44 | 0.53 | 0.48 |
| Перес_тёмное | 0.18 | 0.39 | 0.24 |
| Алевролит | 0.14 | 0.38 | 0.21 |

**Ключевой вывод**: при random split val/test разрыв — 1.07× (было 1.7–2.3× при well-based). **Главная проблема проекта — доменный сдвиг между скважинами**, а не мощность модели. Классы обучаемы; нужна нормализация per-well или больше скважин в обучении.

---

### run_no_merge_dual — ResNet18 Dual, 18 исходных классов

Диагностический эксперимент: обучение без слияния классов (18 оригинальных минералов, порог ≥300 тайлов). Датасет: `data/dataset_no_merge`.

| Модель | Val F1 | Val Acc | Test F1 | Test Acc | Best epoch |
|--------|--------|---------|---------|----------|------------|
| Dual ДС+УФ | 0.242 | 0.293 | 0.185 | 0.183 | 5 ⚠ |

⚠ **Прервано после 6 эпох.**

**Ключевые находки:**
- **Глина_опоковидная_с_включением_глинистых_опок**: F1=**0.98** — исключительно характерная порода, заслуживает отдельного класса
- Все подклассы Перес_светлое: F1≈0 — визуально неразличимы, слияние в один класс обосновано
- Алевролит и все его разновидности: F1 < 0.22 — хронически трудный класс
- Вывод: **текущая таксономия 6 классов в целом правильная**, но Карбонат стоит разбить на Глина_опоковидная + остальной Карбонат

---

## Диагностические инструменты

### prepare_exp.py — создание экспериментальных датасетов

```bash
# 6 классов, случайный интервальный сплит (80/10/10)
python3 prepare_exp.py --mode random

# 18 оригинальных минералов, случайный сплит, порог 300 тайлов
python3 prepare_exp.py --mode no-merge --min-tiles 300
```

Использует существующие тайлы из `data/pipeline/tiles/` — перетайлинг не нужен.

### --dataset-root для train.py и visualize_results.py

```bash
python3 train.py run_name --mode dual --dataset-root data/dataset_random
python3 visualize_results.py run_name --dual --dataset-root data/dataset_random
```

---

## Что нужно сделать

### Приоритет: высокий

- **Дотренировать run_random_dual** — `python3 train.py run_random_dual_v2 --mode dual --dataset-root data/dataset_random` (60 эпох, прервано на 4)
- **Нормализация per-well** — вычислять mean/std каждой скважины перед инференсом; главная причина val/test gap
- **Разбить Карбонат** — выделить Глина_опоковидная в отдельный класс (F1=0.98 в no-merge эксперименте)

### Приоритет: средний

- **Дообучить УФ ResNet50** — `python3 train.py run_rn50_v2 --arch resnet50` (~56 эп., прервана на 4)
- **Дообучить Dual ResNet50** — после полного single: `python3 train.py run_rn50_dual_v2 --arch resnet50 --mode dual --warm-start results/run_rn50_v2`
- **Алевролит F1≈0** — focal γ=3–4, или merge с Аргиллитом (5 классов). Визуально близок к Перес_тёмное

### Приоритет: низкий

- **Больше скважин** — 16 мало для надёжной генерализации
- **Сравнение ResNet18 vs ResNet50** — корректно только после полного обучения обоих

---

## История изменений

### v7 (текущая) — 2026-06-05

- **prepare_exp.py**: диагностический пайплайн — создаёт датасеты с random split или без слияния классов из готовых тайлов
- **--dataset-root**: добавлен в `train.py` и `visualize_results.py` для экспериментальных датасетов
- **dataset_random**: 6 классов, интервальный random split 80/10/10
- **dataset_no_merge**: 18 оригинальных минералов, random split, порог 300 тайлов
- **run_random_dual**: подтверждён доменный сдвиг как главная проблема (val/test gap 1.07× vs 1.7–2.3× ранее)
- **run_no_merge_dual**: подтверждена правильность таксономии 6 классов; Глина_опоковидная выделяется F1=0.98

### v6 — 2026-06-04

- **Единый train.py**: заменяет 4 отдельных скрипта; параметры `--arch`, `--mode`, `--modality`, `--warm-start`
- **run.sh**: оркестрирует полный пайплайн (данные → ДС → УФ → визуализация); `trap '' INT` — Ctrl+C завершает только текущий шаг
- **Корректная остановка**: после Ctrl+C графики строятся через свежий загрузчик (`num_workers=0`) без краша DataLoader
- **visualize_results.py**: фиксы — пропорции фото из реальных пикселей, легенда геолога (группировка слитых классов), адаптивный цвет процентов, confusion matrix в %, полные имена классов в таблице, n= под каждым баром F1
- **Эксперименты**: run_rn50_v1 (ResNet50 single), run_rn50_dual (ResNet50 dual, warm start)
- **git history**: очищена — удалено 2.7 ГБ dangling blob-объектов (`git gc --prune=now`)

### v5 — 2026-06-04

- **ResNet50**: добавлены `create_resnet50`, `DualStreamResNet50` (4096→1024→N); батч RN50=32, dual=16
- **Auto-detect архитектуры**: `visualize_results.py` читает `arch` из `config.json` каждого run
- **Ноутбуки**: помечены как устаревшие (заменены CLI-скриптами)

### v4 — 2026-06-04

- **Мультимодальная модель**: DualStreamResNet18 — два ResNet18 backbone, late fusion cat(512+512)→1024→512→N; синхронный CutMix
- **Таксономия**: Переслаивание разделено на Перес_светлое + Перес_тёмное; Уголь возвращён в Аргиллит (мало данных)
- **Пайплайн**: ноутбуки заменены на CLI-скрипты (`prepare_data.py`, `visualize_results.py`)
- **LR-расписание**: OneCycleLR (pct_start=0.25) вместо ReduceLROnPlateau; best epoch ожидается > 20
- **Loss**: FocalLoss(γ=2, class_weights) — совместная балансировка дисбаланса
- **Visualizer**: автоматическое определение num_classes из checkpoint — поддерживает старые и новые веса

### v3

- **Таксономия**: 8 → 5 классов по принципу визуальной отличимости
- **Тайлинг**: TILE_CM 5 → 10 см, OVERLAP_CM 1 → 0 (убраны квазидубликаты)
- **Нормализация**: per-modality из датасета (DS mean=0.58, UV mean=0.19)
- **Регуляризация**: LabelSmoothing, CutMix, clip_grad, warmup, class_weights вместо WeightedRandomSampler

### v2

- Слит класс "Уголь" с "Аргиллит"
- Стратифицированный жадный сплит по скважинам
- SGD → Adam, CosineAnnealingLR → ReduceLROnPlateau
