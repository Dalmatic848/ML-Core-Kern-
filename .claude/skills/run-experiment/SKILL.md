---
name: run-experiment
description: Запустить новый обучающий эксперимент (train.py + визуализация) как провалидированный, версионируемый прогон вместо разовой команды в терминале. Используй, когда пользователь просит обучить модель, попробовать новую конфигурацию (архитектуру/scheduler/dataset), или прогнать пачку экспериментов.
---

# Запуск эксперимента

Эксперимент в этом проекте — это файл `experiments/<name>.yaml`, а не
одноразовая команда в терминале. Так прогон остаётся воспроизводимым из
чистого клона (`results/`/`logs/` не версионируются — см. `.gitignore` —
но рецепт эксперимента версионируется).

## Шаги

1. **Создай `experiments/<name>.yaml`** по образцу существующих
   (`experiments/13_r18_soft_plateau.yaml` и соседние). Обязательные поля:
   - `name` — должно совпадать с именем файла (без `.yaml`), это
     проверяется тестом `tests/test_experiments_config.py`.
   - `train_args` — список аргументов ровно как в `python3 train.py --help`
     (позиционный `run_name` первым).
   - `visualize_script` / `visualize_args` (опционально) — `visualize_results.py`
     для 6-классовых моделей, `visualize_soft.py` для soft-label.

2. **Один эксперимент**:
   ```bash
   python3 scripts/run_experiment.py experiments/<name>.yaml
   ```
   Смотри `logs/adhoc_<timestamp>/<name>_{train,vis}.log` и `_summary.json`.

3. **Пачка экспериментов** (несколько `experiments/*.yaml` подряд, как
   раньше делал `run_overnight.sh`):
   ```bash
   python3 scripts/run_batch.py                       # все experiments/*.yaml
   python3 scripts/run_batch.py --only <name>          # один
   python3 scripts/run_batch.py --skip 2               # пропустить первые N
   ```
   Успех/провал определяется РЕАЛЬНЫМ кодом возврата процесса (не grep по
   stdout, как раньше) — итоговый ненулевой exit code означает, что хотя бы
   один эксперимент в пачке упал.

4. Результат попадает в `results/<run_name>/` (не версионируется). Если
   эксперимент — лучший на сегодня, обнови таблицу «Лучшие результаты» и
   «Хронологию экспериментов» в README.md вручную, либо запусти агента
   `experiment-analyst`, который подготовит черновик записи.

## Не делай

- Не редактируй `run_overnight.sh` — его больше нет, это устаревший путь.
- Не добавляй эксперимент прямо в `train.py`/`scripts/run_batch.py` —
  только через новый `experiments/*.yaml`.
