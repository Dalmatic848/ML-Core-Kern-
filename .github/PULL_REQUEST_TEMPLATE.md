## Что и зачем



## Чек-лист

- [ ] `ruff check .` и `pytest` проходят локально (CI это тоже проверит)
- [ ] Если менялась таксономия (`src/taxonomy.py`/`soft_labels_config.json`) — `python3 scripts/validate_taxonomy.py` пройден
- [ ] Если добавлен новый лучший эксперимент — README.md («Лучшие результаты», «Хронология экспериментов») обновлён
- [ ] Ничего из `data/`, `results/`, `logs/`, `models/*` не закоммичено случайно
