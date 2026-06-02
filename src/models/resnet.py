import torch.nn as nn
from torchvision import models


def create_resnet18(
    num_classes: int,
    freeze_mode: str = "none",
    dropout_p: float = 0.5,
) -> nn.Module:
    """
    freeze_mode:
        'none'    — fine-tune все слои
        'partial' — заморозить conv1, bn1, layer1, layer2
        'full'    — обучать только layer4 + fc
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if freeze_mode == "partial":
        for name, param in model.named_parameters():
            if any(x in name for x in ["conv1", "bn1", "layer1", "layer2"]):
                param.requires_grad = False
    elif freeze_mode == "full":
        for name, param in model.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout_p),
        nn.Linear(in_features, num_classes),
    )
    return model
