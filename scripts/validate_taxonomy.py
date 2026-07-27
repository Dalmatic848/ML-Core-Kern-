#!/usr/bin/env python3
"""Быстрая проверка целостности таксономии (src/taxonomy.py) — используется
как pre-commit хук и как отдельная ручная проверка после правки
src/taxonomy.py или soft_labels_config.json.

Запуск:
    python3 scripts/validate_taxonomy.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.taxonomy import CLASSES_ORDER, MINERAL_TO_CLASS, load_soft_labels_config, validate_taxonomy


def main() -> int:
    try:
        validate_taxonomy()
    except AssertionError as e:
        print(f"ОШИБКА таксономии: {e}", file=sys.stderr)
        return 1

    base_classes, soft_map = load_soft_labels_config()
    print(f"OK: {len(MINERAL_TO_CLASS)} минералов → {len(CLASSES_ORDER)} классов "
         f"({', '.join(CLASSES_ORDER)}); {len(soft_map)} soft-label записей валидны "
         f"({len(base_classes)} базовых компонент).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
