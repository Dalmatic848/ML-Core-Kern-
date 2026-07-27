"""
Единая точка сборки и загрузки моделей.

Раньше архитектурный dispatch (какую фабрику вызвать для какого arch/mode)
был независимо реализован в train.py (ARCH_REGISTRY), visualize_results.py
(_load_single/_load_dual) и visualize_soft.py (load_model) — три разных
if/elif цепочки, которые нужно было синхронизировать вручную. Здесь один
build_model() для всех троих.

dual resnet18/50 больше не имеют отдельной реализации (src/models/dual_resnet.py
удалён): DualStreamModel из backbones.py собирает идентичную по именам и
формам весов архитектуру (тот же backbone_ds/backbone_uv + head.0-3
Sequential) — старые dual-чекпоинты грузятся в неё без изменений, включая
load_from_single_checkpoints() для warm-start.

load_checkpoint() дополнительно устраняет источник самого запутанного бага
в этом проекте (visual_1m.ipynb: "size mismatch ... [7,512] ... [5,512]"):
раньше num_classes для новой модели брался из ВНЕШНЕГО источника
(label_encoder.json активного датасета), который мог не совпадать с
классами, на которых чекпоинт реально обучен — PyTorch падал с невнятным
shape-mismatch. Теперь num_classes определяется ИЗ САМОГО чекпоинта
(последний Linear-слой head/fc/classifier), а label_encoder.json (если
передан) используется только для явной, понятной проверки согласованности.
"""

from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn

from src.models.backbones import create_dual, create_single
from src.models.resnet import create_resnet18, create_resnet50

_LEGACY_RESNET_ARCHES = {"resnet18", "resnet50"}


def build_model(arch: str, mode: str, num_classes: int,
                freeze_mode: str = "none", dropout_p: float = 0.5) -> nn.Module:
    """Собирает необученную модель для (arch, mode).

    single + resnet18/50 идёт через src/models/resnet.py (голова именуется
    `fc.*` — так же, как во всех уже сохранённых одиночных чекпоинтах).
    Всё остальное (single других архитектур, и ЛЮБОЙ dual) — через
    src/models/backbones.py, которое собирает голову как `head.*`.
    """
    if mode == "single" and arch in _LEGACY_RESNET_ARCHES:
        fn = create_resnet18 if arch == "resnet18" else create_resnet50
        return fn(num_classes, freeze_mode=freeze_mode, dropout_p=dropout_p)
    if mode == "single":
        return create_single(arch, num_classes, freeze_mode=freeze_mode, dropout_p=dropout_p)
    return create_dual(arch, num_classes, freeze_mode=freeze_mode, dropout_p=dropout_p)


def infer_num_classes(state: dict, dual: bool) -> int:
    """Определяет число классов из последнего Linear-слоя чекпоинта, а не из
    внешнего label_encoder.json — так модель всегда собирается под ФАКТИЧЕСКИЙ
    checkpoint, и load_state_dict не падает с shape mismatch."""
    key = "head.4.weight" if dual else "fc.1.weight"
    if key in state:
        return state[key].shape[0]
    for k, v in reversed(list(state.items())):
        if k.endswith(".weight") and v.dim() == 2:
            if any(s in k for s in ("head.", "fc.", "classifier.")):
                return v.shape[0]
    raise ValueError(f"Не удалось определить num_classes из checkpoint. Ключи: {list(state)[:10]}")


def read_arch_from_config(run_dir: Path, default: str = "resnet18") -> str:
    """Архитектура из results/<run>/config.json; 'dual_' префикс (устаревшее
    именование ранних экспериментов) отбрасывается."""
    cfg_path = Path(run_dir) / "config.json"
    if cfg_path.exists():
        import json
        with open(cfg_path, encoding="utf-8") as f:
            return json.load(f).get("arch", default).replace("dual_", "")
    return default


def _label_encoder_class_count(label_encoder_path: Path) -> int:
    import json
    with open(label_encoder_path, encoding="utf-8") as f:
        return len(json.load(f))


def load_checkpoint(
    ckpt_path: Union[str, Path],
    mode: str,
    arch: Optional[str] = None,
    run_dir: Optional[Union[str, Path]] = None,
    label_encoder_path: Optional[Union[str, Path]] = None,
    expected_num_classes: Optional[int] = None,
    freeze_mode: str = "none",
    dropout_p: float = 0.5,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Грузит модель из checkpoint. arch, если не передан, читается из
    run_dir/config.json. Если передан label_encoder_path или
    expected_num_classes, дополнительно проверяет согласованность с числом
    классов в checkpoint — и бросает понятную ошибку вместо PyTorch shape
    mismatch, если это не так (типичная причина: чекпоинт от другого варианта
    датасета/таксономии, чем та, что сейчас активна)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(ckpt_path)

    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = {k.replace("module.", "").replace("model.", ""): v for k, v in state.items()}

    dual = mode == "dual"
    num_classes = infer_num_classes(state, dual=dual)

    if label_encoder_path is not None:
        expected_num_classes = _label_encoder_class_count(Path(label_encoder_path))
    if expected_num_classes is not None and expected_num_classes != num_classes:
        raise ValueError(
            f"Несогласованность классов: checkpoint {ckpt_path} обучен на "
            f"{num_classes} класс(ов), а ожидалось {expected_num_classes}. "
            f"Скорее всего чекпоинт — от другого варианта датасета "
            f"(например, no-merge/soft вместо default). Укажите правильный "
            f"label_encoder.json/dataset-root для этого run."
        )

    if arch is None:
        arch = read_arch_from_config(run_dir) if run_dir is not None else "resnet18"

    model = build_model(arch, mode, num_classes, freeze_mode=freeze_mode, dropout_p=dropout_p)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
