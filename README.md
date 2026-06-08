# ML-Core-Kern — Литотипизация керна

Классификация литотипов горных пород по фотографиям бурового керна.  
Две модальности: **ДС** (дневной свет) и **УФ** (ультрафиолет), тайлы 10 см.

---

## Лучшие результаты на сегодня

| Подход | Арх | Датасет | val F1 | test F1 | Эпох |
|--------|-----|---------|--------|---------|------|
| RN18 dual + plateau *(run_10)* | ResNet18 | random 6кл | **0.611** | **0.553** | 60 |
| ConvNeXt-Tiny dual + plateau *(run_16)* | ConvNeXt-Tiny | random 6кл | 0.606 | 0.552 | 18 |
| RN50 soft labels + plateau *(run_17)* | ResNet50 | soft 5комп | 0.592 | **0.564** | 22 |

**Ключевой вывод**: val/test gap снизился с 1.7–2.3× (well-split) до 1.07–1.10× (random split), что подтвердило — основная проблема была domain shift между скважинами, а не архитектура.

---

## Задача

По фотографии фрагмента керна (тайл 10 см) определить литотип породы.

### Стандартная 6-классовая таксономия

| # | Класс | Описание |
|---|-------|---------|
| 0 | Алевролит | Чистый алевролит и разновидности |
| 1 | Аргиллит | Аргиллит + угольные разности |
| 2 | Карбонат | Глинисто-карбонатные, кремнисто-глинистые, глина опоковидная |
| 3 | Перес_светлое | Переслаивание с прослоями песчаника |
| 4 | Перес_тёмное | Переслаивание аргиллит+алевролит без песчаника |
| 5 | Песчаник | Песчаник и карбонатно-цементированный песчаник |

### Soft Labels таксономия (5 базовых компонент)

Альтернативный подход: каждый исходный минерал → вектор состава `[Пес, Алев, Арг, Карб, Гл_оп]`.  
Пример: `Алевролит_с_прослоями_песчаника` → `[0.25, 0.75, 0, 0, 0]`.  
Подробнее: `prepare_exp.py --mode soft`.

---

## Архитектура

```
ДС-фото → ResNet/EffNet/ConvNeXt → feat_ds ─┐
                                              ├─ concat → MLP head → logits
УФ-фото → ResNet/EffNet/ConvNeXt → feat_uv ─┘
```

Поддерживаемые backbone'ы: `resnet18`, `resnet50`, `efficientnet_b3`, `convnext_tiny`.

---

## Структура проекта

```
ML-Core-Kern-/
├── config.py                    # Пути, гиперпараметры, цвета
├── train.py                     # Единый скрипт обучения
├── visualize_results.py         # Визуализация для 6-классовых моделей
├── visualize_soft.py            # Визуализация для soft-label моделей
├── prepare_exp.py               # Сборка экспериментальных датасетов
├── prepare_data.py              # Основной пайплайн (тайлинг, сплит)
├── run_overnight.sh             # Ночной скрипт последовательного обучения
├── src/
│   ├── models/
│   │   ├── backbones.py         # EfficientNet, ConvNeXt, универсальный DualStreamModel
│   │   ├── dual_resnet.py       # DualStreamResNet18/50
│   │   └── resnet.py            # create_resnet18/50
│   ├── data.py                  # PairedDataset, SoftLabelPairedDataset
│   ├── losses.py                # FocalLoss, LabelSmoothingCE, SoftCrossEntropy
│   ├── transforms.py            # get_transforms()
│   └── training.py              # EarlyStopping, save_history
├── data/
│   ├── dataset/                 # Основной датасет (well-based split)
│   ├── dataset_random/          # Random interval split, 6 классов
│   ├── dataset_soft/            # Soft labels, 30 минералов → 5 компонент
│   └── pipeline/
│       ├── tiles/               # Нарезанные тайлы по скважинам
│       └── csv/                 # Разметка геолога (depth_from, depth_to, mineral)
└── results/                     # Результаты экспериментов (run_01..run_17)
```

---

## Запуск

### Обучение

```bash
source ../venv/bin/activate

# Стандартное (RN18 dual, plateau планировщик, 6 классов)
python3 train.py run_18_test --mode dual \
    --dataset-root data/dataset_random \
    --scheduler plateau --max-epochs 40 --patience 10

# Soft labels (RN50 dual)
python3 train.py run_18_rn50_soft --mode dual --arch resnet50 --soft-labels \
    --dataset-root data/dataset_soft \
    --scheduler plateau --max-epochs 40

# ConvNeXt-Tiny (лучшая скорость сходимости)
python3 train.py run_18_cnxt --mode dual --arch convnext_tiny \
    --dataset-root data/dataset_random \
    --scheduler plateau

# Ночной скрипт (несколько экспериментов подряд)
bash run_overnight.sh
```

### Визуализация

```bash
# 6-классовая модель
python3 visualize_results.py run_10_r18_plateau --dual \
    --dataset-root data/dataset_random

# Soft-label модель
python3 visualize_soft.py --run run_17_r50_soft_plateau

# С TTA и per-well нормализацией
python3 visualize_results.py run_10_r18_plateau --dual \
    --dataset-root data/dataset_random --tta 5 --per-well-norm
```

### Сборка датасета

```bash
# Пересборка random-split (6 классов)
python3 prepare_exp.py --mode random

# Soft labels датасет (порог 50 тайлов)
python3 prepare_exp.py --mode soft --min-tiles 50

# Исходный пайплайн (тайлинг 20 см вместо 10)
python3 prepare_data.py --tile-cm 20 --force
```

---

## Хронология экспериментов

| # | Run | Арх | Данные | Планировщик | val F1 | test F1 | Ключевое изменение |
|---|-----|-----|--------|-------------|--------|---------|-------------------|
| 01 | run | RN18 single | well-split | ReduceLR | 0.320 | ~0.18 | Baseline |
| 02 | run_v2 | RN18 single | well-split | ReduceLR | 0.404 | — | Adam, меньше классов |
| 03 | run_v3 | RN18 single | well-split | ReduceLR | 0.488 | ~0.26 | 10см тайлы, CutMix |
| 04 | run_v4 | RN18 dual | well-split | OneCycle | 0.452 | ~0.28 | Первый dual-stream |
| 05 | run_rn50_v1 | RN50 single | well-split | OneCycle | 0.481 | 0.260 | ResNet50 |
| 06 | run_rn50_dual | RN50 dual | well-split | OneCycle | 0.478 | 0.285 | RN50 warm start |
| 07 | run_no_merge_dual | RN18 dual | 18 классов | OneCycle | 0.242 | 0.185 | Диагностика таксономии |
| 08 | run_random_dual | RN18 dual | random 6кл | OneCycle | 0.569 | 0.531 | **Random split** |
| 09 | run_random_dual_v2 | RN18 dual | random 6кл | OneCycle | 0.569 | 0.530 | Дотренировка (бесполезно) |
| 10 | run_random_dual_plateau | RN18 dual | random 6кл | **Plateau** | **0.611** | **0.553** | **Plateau планировщик** |
| 11 | run_soft_dual | RN18 dual | soft 5комп | OneCycle | 0.639* | 0.529 | **Soft labels** |
| 12 | run_soft_plateau | RN18 dual | soft 5комп | — | — | — | Прерван |
| 13 | run_13_r18_soft_plateau | RN18 dual | soft 5комп | Plateau | 0.556 | — | Soft + plateau |
| 14 | run_14_r50_plateau | RN50 dual | random 6кл | Plateau | — | — | OOM → перезапуск |
| 15 | run_15_effb3_plateau | EffB3 dual | random 6кл | Plateau | — | — | OOM → перезапуск |
| 16 | run_16_convnext_plateau | ConvNeXt-T | random 6кл | Plateau | 0.606 | 0.552 | **ConvNeXt** |
| 17 | run_17_r50_soft_plateau | RN50 dual | soft 5комп | Plateau | 0.592 | **0.564** | RN50 + soft |

*\* val F1 для soft — по dominant-компоненте (argmax), не напрямую сравнимо с 6-классовым*

---

## Анализ: Soft Labels — работает ли?

### Что изменилось

Вместо жёстких меток (6 классов) модель предсказывает **вектор состава** из 5 базовых компонент.

**Плюсы — подтверждены экспериментально:**
- Песчаник: F1 0.76 → **0.90** (+14%)
- Аргиллит: F1 0.48 → **0.75** (+27%)
- Глина_опоковидная: F1 ~0.83 → **0.93** (выведена из ошибочного класса "Карбонат")
- Модель честно показывает неопределённость через **энтропию предсказания**
- Алевролит предсказывается как [37% Пес, 27% Алев, 32% Арг] — геологически корректно

**Минусы и открытые вопросы:**

1. **Macro F1 не растёт** (0.529–0.564 vs 0.553 у 6-классовой). Причина: macro F1 считается по argmax, игнорируя непрерывный вектор состава. Это неверная метрика для soft labels.

2. **Пропорции в маппинге субъективны**. Мы задали `Алевролит_с_прослоями_песчаника = [0.25, 0.75, 0, 0, 0]` экспертно. Если реальные пропорции другие — модель учится на "неправде".

3. **Алевролит неразрешим**. Он физически промежуточный между Пес и Арг. Любая модель правильно предсказывает ~37% Пес + 32% Арг — и это не ошибка, это геология.

4. **Карбонат исчезает** (только 3 тестовых тайла). Старая 6-классовая "Карбонат" включала Глину_опоковидную — принципиально разные породы.

### Рекомендация по soft labels

Soft labels — **правильная постановка** для переходных литотипов (Переслаивания, "с_прослоями"). Для оценки нужна **KL-дивергенция** или Earth Mover's Distance, а не macro F1 по argmax. Среди готовых метрик: median KL Песчаника = 0.28 bit (хорошо), Алевролита = 1.5 bit (ожидаемо плохо).

---

## Ключевые находки

### 1. Domain shift — главная проблема (подтверждено)
Well-split → val/test gap 1.7–2.3×. Random split → gap 1.07–1.10×. Скважины различаются по освещению, качеству сканирования и литологическому облику.

### 2. Планировщик важнее архитектуры
OneCycleLR пикует на эпохе 2 и больше не улучшается. ReduceLROnPlateau даёт стабильный рост: +4.2% val F1 (+2.2% test) при той же архитектуре и данных.

### 3. ConvNeXt-Tiny — самая эффективная архитектура
val F1=0.606 за **18 эпох** (RN18 plateau: 0.611 за 60 эпох). Сходится в 3× быстрее.

### 4. Таксономия "Карбонат" была ошибочной
Глина_опоковидная (siliceous clay, F1=0.93) и Карбонат (F1=0.88) — визуально разные породы. Объединение в один класс "Карбонат" было геологически неверным.

### 5. Алевролит неклассифицируем
Во всех экспериментах F1 Алевролита = 0.19–0.24. Модель правильно видит его как промежуточный материал между Песчаником и Аргиллитом. Решение: либо объединить с Аргиллитом, либо принять неопределённость и показывать пользователю вектор состава.

---

## Что ещё можно попробовать

### Данные
- **Крупные тайлы 20–30 см**: `python3 prepare_data.py --tile-cm 20 --force` — переслаивание лучше видно на длинном фрагменте
- **Per-well нормализация**: вычислять mean/std из тайлов конкретной скважины при инференсе (`--per-well-norm`)
- **Пересмотр таксономии**: выделить Глина_опоковидная как отдельный класс, убрать Алевролит или слить с Аргиллитом

### Обучение
- **Longer training**: ConvNeXt с `--max-epochs 60` не упёрся в потолок за 18 эпох
- **Focal gamma для Алевролита**: `focal_gamma=3-4` усилит штраф за редкий класс
- **Правильная метрика для soft labels**: заменить val_f1 на `1 - median_KL` для честного early stopping

### Инференс
- **TTA**: `--tta 5` даёт +0.001 F1 на текущих моделях; вырастет после полного обучения
- **Ансамблирование**: усреднить logits от 6-классовой + soft-label модели

---

## Конфигурация железа

- GPU: NVIDIA RTX 3050 8 GiB (+ ~400 MiB занято Xorg)
- Эффективно для обучения: ~7.4 GiB
- Batch sizes: RN18 dual=32, RN50 dual=16, EffB3 dual=8, ConvNeXt dual=8
- Рекомендуется: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

---

## История версий

| Версия | Дата | Изменения |
|--------|------|-----------|
| v1 | 02.06.26 | Baseline ResNet18, well-split, 9 классов |
| v2 | 03.06.26 | 10 см тайлы, 5 классов, CutMix, dual-stream |
| v3 | 04.06.26 | ResNet50, FocalLoss, 6 классов, OneCycleLR |
| v4 | 05.06.26 | Диагностика domain shift, random split, prepare_exp.py |
| v5 | 05.06.26 | Soft labels, ReduceLROnPlateau, visualize_soft.py |
| v6 | 08.06.26 | EfficientNet-B3, ConvNeXt-Tiny, run_overnight.sh, нумерация run_NN |
