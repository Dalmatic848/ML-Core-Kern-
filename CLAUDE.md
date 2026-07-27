# ML-Core-Kern-

Литотипизация керна по фотографиям (ДС/УФ, dual-stream). См. README.md для
задачи/архитектуры/результатов — здесь только то, что README не покрывает:
конвенции репозитория и рабочий процесс.

## Обязательные правила

- **Таксономия — только `src/taxonomy.py`.** Маппинг минерал→класс и
  soft-label компоненты определяются ровно в одном месте. Не добавляй
  вторую копию в `prepare_dataset.py`/`visualize_soft.py`/куда-либо ещё —
  именно такое дублирование (3 расходящиеся копии) уже приводило к
  реальному багу при смене 8→6 классов. После правки таксономии:
  `python3 scripts/validate_taxonomy.py` (тот же чек в pre-commit и CI).
- **`results/`, `logs/`, `data/`, `models/*` не версионируются** (см.
  `.gitignore`) — не пытайся их `git add`. Воспроизводимость эксперимента
  обеспечивается версионированием `experiments/*.yaml`, а не сырых
  артефактов.
- **Эксперимент — это файл в `experiments/`**, не правка шелл-скрипта.
  Новый прогон = новый `experiments/<name>.yaml` + `scripts/run_experiment.py`
  или `scripts/run_batch.py`. См. skill `run-experiment`.
- **Работа через PR в `main`**, не прямые коммиты в `main`. См. `CONTRIBUTING.md`.
- Перед коммитом: `ruff check .` и `pytest` должны быть чистыми (это же
  проверяет CI и pre-commit).

## Gotchas (не выводится из кода)

- **Well-based vs random split — не взаимозаменяемы.** Домен (скважина)
  сильно влияет на визуальный вид тайлов (освещение, качество скана).
  Well-based split (`--variant default`, канонический датасет) даёт val/test
  gap 1.7–2.3× — это не переобучение, это domain shift между скважинами.
  Random split (`--variant random`) занижает эту проблему (gap ~1.07–1.10×)
  и подходит только для сравнения архитектур/гиперпараметров между собой,
  не для честной оценки на новых скважинах.
- **"Алевролит" принципиально неразличим** от мелкозернистого песчаника и
  глинистого аргиллита на фото при тайле 10 см — размер зерна не виден
  физически. Стабильно низкий F1 (~0.19–0.24) по этому классу во всех
  экспериментах — не баг модели, это свойство данных (см.
  `soft_labels_config.json:_nota_bene` и README «Ключевые находки»).
- **Граничные тайлы несут ~17–20% шума меток** — первый/последний тайл
  интервала может частично захватывать соседний литотип. Это верхняя
  граница ожидаемого качества, не повод искать баг в пайплайне.
- **GPU 8GB (RTX 3050) — тесная память.** ResNet50/EfficientNet-B3 dual уже
  падали по OOM при неудачных batch size. Используй
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (уже выставлено в
  `scripts/run_batch.py`) и не увеличивай batch size в `train.py:ARCH_REGISTRY`
  без проверки на реальном GPU.
- **Многочасовые прогоны — останавливай мягко.** Ctrl+C (SIGINT) убивает
  DataLoader-воркеров раньше, чем main-процесс успевает сохранить историю.
  Для управляемой остановки между эпохами: `touch /tmp/ml_STOP` или
  `kill -USR1 <pid>` (см. `src/training.py:GracefulStop`).
- **`config.json` не хранит `soft_labels`/`modality`.** `predict.py` и
  `src/models/registry.py` определяют soft vs hard по наличию
  `base_classes.json` в директории модели, а не по ключу конфига — учитывай
  это, если пишешь код, читающий `results/run_*/config.json` напрямую.

## Быстрые команды

```bash
source ../venv/bin/activate
pip install -r requirements-dev.txt   # один раз, для ruff/pytest/pre-commit
ruff check .
pytest
python3 scripts/validate_taxonomy.py
```
