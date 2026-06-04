# Пакет с архитектурами моделей
from .resnet import create_resnet18
from .dual_resnet import create_dual_resnet18, DualStreamResNet18

__all__ = ["create_resnet18", "create_dual_resnet18", "DualStreamResNet18"]
