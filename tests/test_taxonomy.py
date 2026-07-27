"""Таксономия (src/taxonomy.py) — единственный источник маппинга минерал→класс.
Раньше расхождения между тремя копиями этого маппинга уже приводили к
реальному багу (8→6 классов), поэтому эти проверки прогоняются в CI."""

import pytest

from src.taxonomy import (
    CLASSES_ORDER,
    MINERAL_TO_CLASS,
    load_soft_labels_config,
    mineral_to_class,
    normalize_mineral_name,
    validate_taxonomy,
)


def test_validate_taxonomy_passes_on_real_config():
    # Бросает AssertionError при малейшей рассинхронизации — если тест
    # проходит, soft_labels_config.json согласован с MINERAL_TO_CLASS.
    validate_taxonomy()


def test_normalize_mineral_name_converts_spaces():
    assert normalize_mineral_name("Глина аргиллитоподобная") == "Глина_аргиллитоподобная"
    assert normalize_mineral_name(" Песчаник ") == "Песчаник"


def test_mineral_to_class_known_mapping():
    assert mineral_to_class("Песчаник") == "Песчаник"
    assert mineral_to_class("Уголь") == "Аргиллит"  # редкий класс слит в Аргиллит
    assert mineral_to_class("Переслаивание песчаника и алевролита") == "Перес_светлое"


def test_mineral_to_class_excluded_returns_none():
    assert mineral_to_class("Известняк") is None
    assert mineral_to_class("несуществующий_минерал") is None


def test_every_non_excluded_mineral_maps_into_classes_order():
    for mineral, cls in MINERAL_TO_CLASS.items():
        if cls is not None:
            assert cls in CLASSES_ORDER, f"{mineral!r} -> {cls!r} не входит в CLASSES_ORDER"


def test_soft_label_vectors_sum_to_one():
    base_classes, soft_map = load_soft_labels_config()
    for mineral, vec in soft_map.items():
        assert len(vec) == len(base_classes), mineral
        assert all(v >= 0 for v in vec), mineral
        assert sum(vec) == pytest.approx(1.0, abs=1e-6), f"{mineral}: {vec}"
