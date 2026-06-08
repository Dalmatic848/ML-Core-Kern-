from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0,
                 weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.alpha  = alpha
        self.gamma  = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return (self.alpha * (1 - pt) ** self.gamma * ce).mean()


class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n = logits.size(1)
        log_probs = F.log_softmax(logits, dim=1)
        one_hot = torch.zeros_like(log_probs).scatter(1, targets.unsqueeze(1), 1)
        smooth = one_hot * (1 - self.smoothing) + (1 - one_hot) * self.smoothing / (n - 1)
        return -(smooth * log_probs).sum(dim=1).mean()


class SoftCrossEntropy(nn.Module):
    """Кросс-энтропия с мягкими метками (распределением вместо int).

    targets — float тензор (B, C) с вероятностями, сумма по dim=1 ≈ 1.
    Совместима с CutMix: можно передавать lam*soft_a + (1-lam)*soft_b.
    """

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        return -(targets * log_probs).sum(dim=1).mean()


def mean_kl_divergence(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """KL(targets || predicted) — среднее по батчу, в битах.

    Чем меньше, тем точнее предсказан состав. 0 = идеал.
    Используется как метрика качества (не для градиентов).
    """
    pred_probs = F.softmax(logits, dim=1).clamp(min=1e-9)
    true_probs = targets.clamp(min=1e-9)
    kl = (true_probs * (true_probs.log() - pred_probs.log())).sum(dim=1)
    return kl.mean() / torch.log(torch.tensor(2.0))  # в битах


def get_criterion(
    loss_type: str = "ce",
    class_weights: Optional[torch.Tensor] = None,
    focal_gamma: float = 2.0,
) -> nn.Module:
    if loss_type == "focal":
        return FocalLoss(gamma=focal_gamma, weight=class_weights)
    if loss_type == "label_smooth":
        return LabelSmoothingCE(smoothing=0.1)
    if loss_type == "soft_ce":
        return SoftCrossEntropy()
    return nn.CrossEntropyLoss(weight=class_weights)
