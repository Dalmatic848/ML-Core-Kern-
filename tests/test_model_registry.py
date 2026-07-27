"""src/models/registry.py — единая точка сборки/загрузки моделей. Покрывает
в первую очередь риск от удаления src/models/dual_resnet.py: старые
dual-чекпоинты и warm-start из одиночных чекпоинтов должны продолжать
работать через общую DualStreamModel (backbones.py). Без GPU, без реальных
данных — только маленькие синтетические тензоры/чекпоинты."""

import pytest
import torch

from src.models.registry import build_model, infer_num_classes, load_checkpoint


def test_build_model_single_resnet18_output_shape():
    model = build_model("resnet18", "single", num_classes=6)
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert out.shape == (2, 6)


def test_build_model_dual_resnet18_output_shape():
    model = build_model("resnet18", "dual", num_classes=6)
    x_ds = torch.randn(2, 3, 64, 64)
    x_uv = torch.randn(2, 3, 64, 64)
    out = model(x_ds, x_uv)
    assert out.shape == (2, 6)


def test_infer_num_classes_from_dual_state_dict():
    model = build_model("resnet18", "dual", num_classes=6)
    assert infer_num_classes(model.state_dict(), dual=True) == 6


def test_infer_num_classes_from_single_state_dict():
    model = build_model("resnet50", "single", num_classes=3)
    assert infer_num_classes(model.state_dict(), dual=False) == 3


def test_load_checkpoint_round_trip_dual(tmp_path):
    model = build_model("resnet18", "dual", num_classes=6)
    ckpt_path = tmp_path / "dual_best.pth"
    torch.save(model.state_dict(), ckpt_path)

    loaded = load_checkpoint(ckpt_path, mode="dual", arch="resnet18",
                             device=torch.device("cpu"))
    x_ds, x_uv = torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64)
    assert loaded(x_ds, x_uv).shape == (1, 6)


def test_warm_start_from_single_checkpoint_into_dual_model(tmp_path):
    """Регрессионный тест на удаление dual_resnet.py: одиночный чекпоинт
    (как сохраняет train.py для --mode single) должен грузиться в backbone
    общей DualStreamModel без ошибок формы."""
    single = build_model("resnet18", "single", num_classes=6)
    ckpt_path = tmp_path / "single_best.pth"
    torch.save(single.state_dict(), ckpt_path)

    dual = build_model("resnet18", "dual", num_classes=6)
    dual.load_from_single_checkpoints(ds_ckpt_path=ckpt_path, uv_ckpt_path=ckpt_path,
                                      device=torch.device("cpu"))
    # Не должно упасть; выход по-прежнему нужной формы.
    out = dual(torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64))
    assert out.shape == (1, 6)


def test_load_checkpoint_raises_clear_error_on_class_mismatch(tmp_path):
    model = build_model("resnet18", "dual", num_classes=6)
    ckpt_path = tmp_path / "dual_best.pth"
    torch.save(model.state_dict(), ckpt_path)

    with pytest.raises(ValueError, match="Несогласованность классов"):
        load_checkpoint(ckpt_path, mode="dual", arch="resnet18",
                        expected_num_classes=5, device=torch.device("cpu"))
