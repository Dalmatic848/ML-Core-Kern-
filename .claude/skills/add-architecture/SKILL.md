---
name: add-architecture
description: Зарегистрировать новую backbone-архитектуру (например новую torchvision-модель) для single/dual-stream обучения. Используй, когда пользователь просит добавить новую архитектуру/backbone к train.py или сравнить with текущими resnet18/50, efficientnet_b3, convnext_tiny.
---

# Добавление новой архитектуры

Все архитектуры проходят через единый `src/models/registry.py`
(`build_model`) — это заменило три независимых arch-dispatch, которые
раньше были в `train.py`, `visualize_results.py` и `visualize_soft.py`.
Не создавай четвёртую копию dispatch-логики — расширяй только
`src/models/backbones.py` + `ARCH_REGISTRY` в `train.py`.

## Шаги

1. **`src/models/backbones.py`** — добавь feature extractor:
   - Добавь `'<arch>': <feat_dim>` в `FEAT_DIMS`.
   - В `_make_extractor()` добавь ветку, возвращающую backbone без головы
     (см. `_EfficientNetExtractor`/`_ConvNeXtExtractor` как образцы для
     моделей, где нельзя просто занулить `.fc`).
   - `SingleStreamModel`/`DualStreamModel` подхватят новую архитектуру
     автоматически — они уже используют `_make_extractor`/`FEAT_DIMS`
     дженерик.

2. **`train.py:ARCH_REGISTRY`** — добавь `('<arch>', 'single')` и
   `('<arch>', 'dual')` через `_make_arch_entry('<arch>', mode, batch_size, lr)`.
   Batch size подбирается под доступную GPU-память (8GB RTX 3050 — см.
   `CLAUDE.md`, gotcha про OOM) — начни с малого batch и увеличивай,
   проверяя реальным прогоном:
   ```bash
   python3 train.py smoke_test --arch <arch> --mode dual --max-epochs 1
   ```

3. **Тест**: добавь в `tests/test_model_registry.py` аналог
   `test_build_model_dual_resnet18_output_shape`, но для новой архитектуры —
   быстрая проверка формы выхода без GPU и без реальных данных.

4. `visualize_results.py`/`visualize_soft.py`/`predict.py` ничего не нужно
   трогать — они уже строят модель через `src.models.registry.build_model`/
   `load_checkpoint`, который читает `arch` из `config.json`.

## Не делай

- Не добавляй новый класс типа `DualStreamXxxNet` по образцу старого
  `src/models/dual_resnet.py` — этот файл был удалён именно потому, что
  дублировал `DualStreamModel` из `backbones.py`. Для новых архитектур
  всегда используй generic `create_single`/`create_dual`.
