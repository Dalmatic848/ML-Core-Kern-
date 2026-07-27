"""
Инференс на одном уже нарезанном тайле — используя promoted-модель из
models/<name>/ (см. models/README.md про конвенцию promoted-моделей).

Раньше инференс-логика была размазана внутри visualize_results.py/
visualize_soft.py (там она встроена в визуализацию целых фотографий на
1 метр) — не было отдельного пути "предскажи класс для одного тайла".

Вход — уже нарезанный тайл нужной высоты (TILE_CM см из config.py), а не
сырое фото керна на 1 метр: калибровка "см на пиксель" для произвольного
нового фото требует знания глубины интервала, которого при чистом
инференсе на новом материале обычно ещё нет. Если нужно проходить по
целой фотографии — см. visualize_results.py/_predict_single/_predict_dual.

Запуск:
    python3 predict.py --model-dir models/production --image tile_ds.jpg --uv-image tile_uv.jpg
    python3 predict.py --model-dir models/production_single --image tile.jpg
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from src.models.registry import load_checkpoint
from src.transforms import get_transforms


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def predict(model_dir: Path, image_path: Path, uv_image_path: Path = None,
           modality: str = "ДС") -> dict:
    cfg = _load_json(model_dir / "config.json")
    mode = "dual" if (cfg.get("mode") == "dual" or uv_image_path is not None) else "single"
    # config.json не записывает soft_labels — используем наличие
    # base_classes.json (копируется в models/<name>/ только для soft-моделей,
    # см. models/README.md) как явный, надёжный сигнал.
    soft = (model_dir / "base_classes.json").exists()

    norm = _load_json(model_dir / "normalization_stats.json")

    if soft:
        classes = _load_json(model_dir / "base_classes.json")
    else:
        label_encoder = _load_json(model_dir / "label_encoder.json")
        classes = [None] * len(label_encoder)
        for name, idx in label_encoder.items():
            classes[idx] = name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if mode == "dual":
        ckpt_path = model_dir / "dual_best.pth"
        model = load_checkpoint(ckpt_path, mode="dual", run_dir=model_dir,
                                expected_num_classes=len(classes), device=device)
        ds_tfm = get_transforms("square", "none", False, norm["ДС"]["mean"], norm["ДС"]["std"])
        uv_tfm = get_transforms("square", "none", False, norm["УФ"]["mean"], norm["УФ"]["std"])
        img_ds = ds_tfm(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        img_uv = uv_tfm(Image.open(uv_image_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(img_ds, img_uv)
    else:
        ckpt_path = model_dir / f"{modality}_best.pth"
        model = load_checkpoint(ckpt_path, mode="single", run_dir=model_dir,
                                expected_num_classes=len(classes), device=device)
        tfm = get_transforms("square", "none", False, norm[modality]["mean"], norm[modality]["std"])
        img = tfm(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(img)

    probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    top_idx = int(probs.argmax())
    return {
        "class": classes[top_idx],
        "confidence": float(probs[top_idx]),
        "probabilities": {c: float(p) for c, p in zip(classes, probs)},
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", required=True, type=Path,
                   help="Директория promoted-модели, см. models/README.md")
    p.add_argument("--image", required=True, type=Path, help="Тайл ДС (или единственной модальности)")
    p.add_argument("--uv-image", type=Path, default=None, help="Тайл УФ (только для dual-моделей)")
    p.add_argument("--modality", default="ДС", choices=["ДС", "УФ"],
                   help="Только для single-режима: какой чекпоинт использовать")
    return p.parse_args()


def main():
    args = parse_args()
    result = predict(args.model_dir, args.image, args.uv_image, args.modality)
    print(f"Класс      : {result['class']}")
    print(f"Уверенность: {result['confidence']:.1%}")
    print("Все вероятности:")
    for cls, p in sorted(result["probabilities"].items(), key=lambda kv: -kv[1]):
        print(f"  {cls:<20} {p:.1%}")


if __name__ == "__main__":
    main()
