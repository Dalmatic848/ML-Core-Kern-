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

## Текущие результаты

### run_v3 — Single-stream, 5 классов (baseline)

| Модальность | F1 macro | Acc | Best epoch |
|-------------|----------|-----|------------|
| ДС | **0.488** | 0.712 | 6 |
| УФ | **0.418** | 0.643 | 7 |

**F1 по классам (val):**

| Класс | ДС | УФ |
|-------|----|----|
| Переслаивание | 0.784 | 0.753 |
| Песчаник | 0.757 | 0.573 |
| Аргиллит | 0.447 | 0.581 |
| Карбонат | 0.452 | 0.168 |
| Алевролит | 0.000 | 0.016 |

Главные проблемы: Алевролит F1≈0, Карбонат УФ F1=0.17, best epoch 6–7 из 60 (модель останавливалась слишком рано из-за ReduceLROnPlateau).

---

### run_v4 — Dual-stream, 6 классов (текущий)

| Модель | F1 macro | Acc | Best epoch |
|--------|----------|-----|------------|
| Dual ДС+УФ | **0.452** | 0.610 | 2 |

**F1 по классам (val):**

| Класс | Dual |
|-------|------|
| Перес_светлое | **0.79** |
| Песчаник | **0.77** |
| Аргиллит | 0.56 |
| Карбонат | 0.35 |
| Перес_тёмное | 0.19 |
| Алевролит | 0.06 |

**Ключевые наблюдения:**
1. **Разделение Переслаивания дало результат**: Перес_светлое F1=0.79 — лучший показатель за всё время. Смешанный класс был источником ошибок.
2. **Dual-поток улучшил Аргиллит** (0.56 vs 0.45 в ДС run_v3) — УФ добавляет диагностическую информацию.
3. **Best epoch = 2** — модель останавливается слишком рано. Датасет перестраивался в процессе (7→6 классов), и модель обучалась на неполной версии. Нужен чистый запуск после `prepare_data.py --force`.
4. **Алевролит по-прежнему F1≈0** — класс слишком редкий (~1 200 train-тайлов) и похож на Аргиллит и Перес_тёмное.
5. **Карбонат в УФ** — ДС модель (run_v3) давала 0.45, dual 0.35. Требует отдельного внимания.

---

## Что нужно сделать

### Приоритет: высокий

- **Пересборка датасета** — `python3 prepare_data.py --force` с актуальными 6 классами (сейчас label_encoder.json обновлён, но tiles/ ещё содержит Уголь папки)
- **Переобучение** — `python3 train_multimodal.py run_dual_v2` после пересборки. Ожидаем best epoch > 20
- **Тёплый старт** — сначала обучить single-stream (`train_run.py run_v5`), затем dual с warm start из этих весов

### Приоритет: средний

- **Алевролит**: focal loss с γ=3-4 специально для него, или объединить с Аргиллитом (4→5 классов)
- **Перес_тёмное F1=0.19**: 1 859 тайлов — достаточно, но визуально похож на Аргиллит. Попробовать per-class focal
- **ResNet50 эксперимент** — `train_run_rn50.py` (два single) + `train_multimodal_rn50.py` (один dual), сравнить с ResNet18

### Приоритет: низкий

- **Нормализация by-well** — mean/std из конкретной скважины перед подачей
- **Больше скважин** — 16 скважин критически мало для надёжного test-gap

---

## История изменений

### v5 (текущая) — 2026-06-04

- **ResNet50 эксперимент**: добавлены `create_resnet50`, `DualStreamResNet50`, скрипты `train_run_rn50.py` и `train_multimodal_rn50.py`
- **Батч-размеры**: RN50 single=32, RN50 dual=16 (в 2× меньше из-за 2048-dim признаков)
- **Auto-detect архитектуры**: `visualize_results.py` читает `arch` из `config.json` каждого run
- **Ноутбуки**: помечены как устаревшие артефакты в структуре проекта

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
