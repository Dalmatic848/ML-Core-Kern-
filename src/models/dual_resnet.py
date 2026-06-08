from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models


def _make_backbone(freeze_mode: str = 'none') -> nn.Module:
    """ResNet18 без fc-головы. Выход: (B, 512) после avgpool."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    if freeze_mode == 'partial':
        for name, param in model.named_parameters():
            if any(x in name for x in ['conv1', 'bn1', 'layer1', 'layer2']):
                param.requires_grad = False
    elif freeze_mode == 'full':
        for name, param in model.named_parameters():
            if 'layer4' not in name and 'fc' not in name:
                param.requires_grad = False
    model.fc = nn.Identity()
    return model


class DualStreamResNet18(nn.Module):
    """Два ResNet18 backbone (ДС + УФ) → late fusion → классификатор.

    Forward: (x_ds, x_uv) → logits (B, num_classes)
    """

    def __init__(
        self,
        num_classes: int,
        freeze_mode: str = 'none',
        dropout_p: float = 0.5,
    ) -> None:
        super().__init__()
        self.backbone_ds = _make_backbone(freeze_mode)
        self.backbone_uv = _make_backbone(freeze_mode)
        self.head = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(512, num_classes),
        )

    def forward(self, x_ds: torch.Tensor, x_uv: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([self.backbone_ds(x_ds), self.backbone_uv(x_uv)], dim=1)
        return self.head(feat)

    def load_from_single_checkpoints(
        self,
        ds_ckpt_path: Optional[Path] = None,
        uv_ckpt_path: Optional[Path] = None,
        device: torch.device = torch.device('cpu'),
    ) -> None:
        """Загружает backbone-веса из одиночных checkpoint'ов (run_v4).
        Ключи fc.* пропускаются — голова имеет другую размерность."""
        if ds_ckpt_path and Path(ds_ckpt_path).exists():
            state = torch.load(ds_ckpt_path, map_location=device, weights_only=True)
            backbone_state = {k: v for k, v in state.items() if not k.startswith('fc.')}
            self.backbone_ds.load_state_dict(backbone_state, strict=False)
            print(f'  backbone_ds ← {ds_ckpt_path}')
        if uv_ckpt_path and Path(uv_ckpt_path).exists():
            state = torch.load(uv_ckpt_path, map_location=device, weights_only=True)
            backbone_state = {k: v for k, v in state.items() if not k.startswith('fc.')}
            self.backbone_uv.load_state_dict(backbone_state, strict=False)
            print(f'  backbone_uv ← {uv_ckpt_path}')


def _make_backbone_rn50(freeze_mode: str = 'none') -> nn.Module:
    """ResNet50 без fc-головы. Выход: (B, 2048) после avgpool."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    if freeze_mode == 'partial':
        for name, param in model.named_parameters():
            if any(x in name for x in ['conv1', 'bn1', 'layer1', 'layer2']):
                param.requires_grad = False
    elif freeze_mode == 'full':
        for name, param in model.named_parameters():
            if 'layer4' not in name and 'fc' not in name:
                param.requires_grad = False
    model.fc = nn.Identity()
    return model


class DualStreamResNet50(nn.Module):
    """Два ResNet50 backbone (ДС + УФ) → late fusion → классификатор.

    Forward: (x_ds, x_uv) → logits (B, num_classes)
    Fusion: cat(2048 + 2048) = 4096 → 1024 → num_classes
    """

    def __init__(
        self,
        num_classes: int,
        freeze_mode: str = 'none',
        dropout_p: float = 0.5,
    ) -> None:
        super().__init__()
        self.backbone_ds = _make_backbone_rn50(freeze_mode)
        self.backbone_uv = _make_backbone_rn50(freeze_mode)
        self.head = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x_ds: torch.Tensor, x_uv: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([self.backbone_ds(x_ds), self.backbone_uv(x_uv)], dim=1)
        return self.head(feat)

    def load_from_single_checkpoints(
        self,
        ds_ckpt_path: Optional[Path] = None,
        uv_ckpt_path: Optional[Path] = None,
        device: torch.device = torch.device('cpu'),
    ) -> None:
        if ds_ckpt_path and Path(ds_ckpt_path).exists():
            state = torch.load(ds_ckpt_path, map_location=device, weights_only=True)
            backbone_state = {k: v for k, v in state.items() if not k.startswith('fc.')}
            self.backbone_ds.load_state_dict(backbone_state, strict=False)
            print(f'  backbone_ds ← {ds_ckpt_path}')
        if uv_ckpt_path and Path(uv_ckpt_path).exists():
            state = torch.load(uv_ckpt_path, map_location=device, weights_only=True)
            backbone_state = {k: v for k, v in state.items() if not k.startswith('fc.')}
            self.backbone_uv.load_state_dict(backbone_state, strict=False)
            print(f'  backbone_uv ← {uv_ckpt_path}')


def create_dual_resnet50(
    num_classes: int,
    freeze_mode: str = 'none',
    dropout_p: float = 0.5,
) -> DualStreamResNet50:
    return DualStreamResNet50(num_classes, freeze_mode, dropout_p)


def create_dual_resnet18(
    num_classes: int,
    freeze_mode: str = 'none',
    dropout_p: float = 0.5,
) -> DualStreamResNet18:
    return DualStreamResNet18(num_classes, freeze_mode, dropout_p)
