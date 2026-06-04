from typing import Tuple

import numpy as np
import torch


def rand_bbox(size: torch.Size, lam: float) -> Tuple[int, int, int, int]:
    W, H = size[2], size[3]
    cut_w = int(W * np.sqrt(1.0 - lam))
    cut_h = int(H * np.sqrt(1.0 - lam))
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1 = int(np.clip(cx - cut_w // 2, 0, W))
    bby1 = int(np.clip(cy - cut_h // 2, 0, H))
    bbx2 = int(np.clip(cx + cut_w // 2, 0, W))
    bby2 = int(np.clip(cy + cut_h // 2, 0, H))
    return bbx1, bby1, bbx2, bby2


def cutmix_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Возвращает (смешанный x, y_a, y_b, lambda_adj)."""
    r = np.random.beta(alpha, alpha)
    lam = max(r, 1 - r)
    idx = torch.randperm(x.size(0), device=x.device)
    x_b, y_b = x[idx], y[idx]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x = x.clone()
    x[:, :, bbx1:bbx2, bby1:bby2] = x_b[:, :, bbx1:bbx2, bby1:bby2]
    lam_adj = 1.0 - (bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2))
    return x, y, y_b, lam_adj


def cutmix_data_paired(
    x_ds: torch.Tensor,
    x_uv: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Синхронный CutMix: одна bbox-маска применяется к обоим потокам.
    Возвращает (x_ds_mixed, x_uv_mixed, y_a, y_b, lambda_adj)."""
    r = np.random.beta(alpha, alpha)
    lam = max(r, 1 - r)
    idx = torch.randperm(x_ds.size(0), device=x_ds.device)
    x_ds_b, x_uv_b, y_b = x_ds[idx], x_uv[idx], y[idx]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x_ds.size(), lam)
    x_ds = x_ds.clone()
    x_uv = x_uv.clone()
    x_ds[:, :, bbx1:bbx2, bby1:bby2] = x_ds_b[:, :, bbx1:bbx2, bby1:bby2]
    x_uv[:, :, bbx1:bbx2, bby1:bby2] = x_uv_b[:, :, bbx1:bbx2, bby1:bby2]
    lam_adj = 1.0 - (bbx2 - bbx1) * (bby2 - bby1) / (x_ds.size(-1) * x_ds.size(-2))
    return x_ds, x_uv, y, y_b, lam_adj
