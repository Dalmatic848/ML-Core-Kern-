# models/ — promoted-чекпоинты

`results/run_*/` содержит все экспериментальные прогоны (см. README.md —
хронология 17 экспериментов). Это большой, шумный, локальный архив — не то,
что нужно колллеге, который просто хочет классифицировать новое фото керна.

`models/` — небольшая, отобранная вручную коллекция: сюда **осознанно
копируется** (не symlink) лучший на сегодня чекпоинт(ы), когда есть решение
использовать их для реального инференса (`predict.py`).

## Конвенция promoted-модели

```
models/<name>/
├── config.json               # скопировано из results/run_.../config.json
├── label_encoder.json        # скопировано из соответствующего data/dataset*/
├── normalization_stats.json  # скопировано оттуда же
├── dual_best.pth             # или {ДС,УФ}_best.pth для single-режима
├── soft_labels.json          # только для soft-label моделей
└── base_classes.json         # только для soft-label моделей
```

Промоутить модель — значит скопировать ВСЕ эти файлы, не только `.pth`.
`predict.py` не обращается к `data/dataset*/` — promoted-модель должна быть
самодостаточной и работать даже на машине коллеги, где `data/` вообще нет.

Пример промоушена (после того как эксперимент признан лучшим):

```bash
mkdir -p models/production
cp results/run_16_convnext_plateau/config.json models/production/
cp results/run_16_convnext_plateau/dual_best.pth models/production/
cp data/dataset_random/label_encoder.json models/production/
cp data/dataset_random/normalization_stats.json models/production/
```

## Инференс

```bash
python3 predict.py --model-dir models/production --image tile_ds.jpg --uv-image tile_uv.jpg
```

См. `predict.py --help` для деталей и ограничений (вход — уже нарезанный
тайл, а не сырое фото керна на 1 метр).
