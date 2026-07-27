---
name: rebuild-dataset-variant
description: Пересобрать или создать новый вариант датасета (well-based, random-split, no-merge, soft labels) через prepare_dataset.py. Используй, когда пользователь просит пересобрать датасет, изменить размер тайла, порог редких классов, или сделать новую комбинацию сплита/таксономии.
---

# Пересборка варианта датасета

`prepare_dataset.py` — единая точка входа (заменил `prepare_data.py` +
`prepare_exp.py`). Один флаг `--variant` выбирает, какой из двух
независимых путей использовать — они НЕ объединены в одну реализацию,
см. `prepare_dataset.py` module docstring, почему.

## Варианты

| `--variant` | Классы | Сплит | Директория по умолчанию |
|---|---|---|---|
| `default` | 6 (слияние) | по скважинам | `data/dataset` |
| `random` | 6 (слияние) | случайный по интервалам | `data/dataset_random` |
| `no-merge` | оригинальные минералы | случайный | `data/dataset_no_merge` |
| `no-merge-well` | оригинальные минералы | по скважинам | `data/dataset_no_merge_well` |
| `soft` | 5 базовых компонент (soft labels) | случайный (или `--split-mode well`) | `data/dataset_soft` |

## Шаги

1. Тайлы (`data/pipeline/tiles/`) собираются один раз, стадии 1-2
   (`--variant default` по умолчанию их включает). Если тайлы уже есть —
   не пересобирай их без необходимости, это долго:
   ```bash
   python3 prepare_dataset.py --stages 3,4,5,6,7    # канонический, без тайлинга
   python3 prepare_dataset.py --variant random        # варианты тайлинг не трогают вообще
   ```

2. Изменение размера тайла — это ПЕРЕТАЙЛИНГ (стадия 2), затрагивает ВСЕ
   варианты (все они читают из одного `data/pipeline/tiles/`):
   ```bash
   python3 prepare_dataset.py --tile-cm 20 --force   # осторожно: удаляет и пересобирает tile_root
   ```

3. Новый минерал в CSV, которого нет в таксономии — `stage2_tile` выведет
   `[WARN] неизвестные минералы: {...}`. Добавь его в
   `src/taxonomy.py:MINERAL_TO_CLASS` (и, если нужно, в
   `soft_labels_config.json:minerals`), затем
   `python3 scripts/validate_taxonomy.py` перед пересборкой.

4. После пересборки датасет содержит `label_encoder.json` +
   `normalization_stats.json` (+ `soft_labels.json`/`base_classes.json` для
   soft) — это то, что `train.py --dataset-root <dir>` и `predict.py`
   ожидают найти.

## Не делай

- Не редактируй маппинг минерал→класс где-либо кроме `src/taxonomy.py`
  (см. `CLAUDE.md`).
- Не путай `--min-tiles` для `default` (мягкий порог на train/val/test,
  см. `_effective_min` в `split_well_greedy`) и для остальных вариантов
  (жёсткий порог, класс с меньшим числом тайлов отбрасывается целиком в
  `filter_classes`) — это разные механики, не опечатка.
