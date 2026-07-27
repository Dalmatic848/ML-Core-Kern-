# Пакет с архитектурами моделей.
# Сборка/загрузка моделей теперь централизована в src/models/registry.py
# (build_model/load_checkpoint) — используйте его вместо прямых импортов
# фабрик ниже, если не пишете код внутри самого пакета models/.
from .backbones import create_dual, create_single
from .resnet import create_resnet18, create_resnet50

__all__ = ["create_resnet18", "create_resnet50", "create_single", "create_dual"]
