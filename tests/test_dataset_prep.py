"""prepare_dataset.py — сборка датасета на синтетических тайлах (без реальных
фото/CSV, без GPU). Проверяет оба независимых пути: канонический
(scan_tiles_by_directory + split_well_greedy, по структуре директорий) и
вариантный (scan_tiles_by_interval + split_random/split_well_based, по
имени файла) — см. модуль docstring prepare_dataset.py про то, почему они
не унифицированы в один алгоритм."""

from PIL import Image

from prepare_dataset import (
    assign_classes,
    build_dataset_from_intervals,
    filter_classes,
    scan_tiles_by_directory,
    scan_tiles_by_interval,
    split_random,
    split_well_based,
    split_well_greedy,
)

CLASSES = ["Песчаник", "Аргиллит", "Алевролит"]
WELLS = ["well_a", "well_b", "well_c"]
MODS = ["ДС", "УФ"]


def _make_fake_tiles(root, n_per_class=20):
    # Имя файла кодирует СЫРОЙ минерал (тут — совпадает с именем класса, для
    # простоты фикстуры); scan_tiles_by_interval группирует по
    # well+mineral+depth из имени файла, а не по директории — поэтому разным
    # классам нужны разные имена минерала, иначе они схлопнутся в один interval.
    for w in WELLS:
        for m in MODS:
            for c in CLASSES:
                d = root / w / m / c
                d.mkdir(parents=True, exist_ok=True)
                for i in range(n_per_class):
                    img = Image.new("RGB", (10, 10), color=(i % 255, 0, 0))
                    img.save(d / f"{c}_1.000_2.000_frag{i:04d}.jpg")


def test_scan_and_split_well_greedy(tmp_path):
    _make_fake_tiles(tmp_path)
    wcc = scan_tiles_by_directory(tmp_path, CLASSES)
    assert set(wcc.keys()) == set(WELLS)

    well_to_split, split_counts = split_well_greedy(
        wcc, CLASSES, val_frac=0.2, test_frac=0.2, seed=42, min_class_tiles=5,
    )
    assert set(well_to_split.values()) <= {"train", "val", "test"}
    assert set(well_to_split.keys()) == set(WELLS)
    # Каждая скважина целиком идёт в один сплит — well-based, не по тайлам.
    assert len(set(well_to_split.values())) >= 2


def test_scan_by_interval_and_random_split(tmp_path):
    _make_fake_tiles(tmp_path, n_per_class=10)
    intervals = scan_tiles_by_interval(tmp_path)
    # 3 скважины x 3 (сырых) минерала x 1 интервал глубины = 9 интервалов.
    assert len(intervals) == 9

    intervals = assign_classes(intervals, no_merge=True)
    intervals, classes_order = filter_classes(intervals, min_tiles=5)
    assert classes_order == sorted(CLASSES)  # сырые имена минералов, no_merge=True

    uid_to_split = split_random(intervals, classes_order, val_frac=0.2, test_frac=0.2, seed=42)
    assert set(uid_to_split.values()) <= {"train", "val", "test"}
    assert set(uid_to_split.keys()) == set(intervals.keys())


def test_filter_classes_drops_rare_classes(tmp_path):
    intervals = {
        "well_a||rare_1_2": {"mineral": "rare", "class_name": "rare",
                             "files": {"ДС": ["f1.jpg"]}},
        "well_a||common_1_2": {"mineral": "common", "class_name": "common",
                               "files": {"ДС": ["f2.jpg"] * 50}},
    }
    filtered, kept = filter_classes(intervals, min_tiles=10)
    assert kept == ["common"]
    assert "well_a||rare_1_2" not in filtered
    assert "well_a||common_1_2" in filtered


def test_split_well_based_keeps_whole_well_together():
    intervals = {
        "well_a||m_1_2": {"well": "well_a", "class_name": "Песчаник", "files": {"ДС": ["a"] * 20}},
        "well_a||m_3_4": {"well": "well_a", "class_name": "Аргиллит", "files": {"ДС": ["b"] * 20}},
        "well_b||m_1_2": {"well": "well_b", "class_name": "Песчаник", "files": {"ДС": ["c"] * 20}},
    }
    uid_to_split = split_well_based(intervals, ["Песчаник", "Аргиллит"],
                                    val_frac=0.3, test_frac=0.3, min_class_tiles=1)
    # Оба интервала well_a должны попасть в один и тот же сплит.
    assert uid_to_split["well_a||m_1_2"] == uid_to_split["well_a||m_3_4"]


def test_build_dataset_from_intervals_writes_label_encoder(tmp_path):
    src_dir = tmp_path / "src_tiles"
    (src_dir).mkdir()
    fake_file = src_dir / "tile.jpg"
    Image.new("RGB", (5, 5)).save(fake_file)

    intervals = {"well_a||m_1_2": {"class_name": "Песчаник", "files": {"ДС": [fake_file]}}}
    uid_to_split = {"well_a||m_1_2": "train"}
    out_dir = tmp_path / "out"

    build_dataset_from_intervals(intervals, uid_to_split, ["Песчаник", "Аргиллит"], out_dir)

    assert (out_dir / "ДС" / "train" / "Песчаник" / "tile.jpg").exists()
    import json
    label_encoder = json.load(open(out_dir / "label_encoder.json", encoding="utf-8"))
    assert label_encoder == {"Песчаник": 0, "Аргиллит": 1}
