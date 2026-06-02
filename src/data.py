from pathlib import Path
from typing import Callable, Optional, Tuple, List

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms


def make_weighted_sampler(
    targets: List[int],
    generator: Optional[torch.Generator] = None,
) -> WeightedRandomSampler:
    class_counts = torch.bincount(torch.tensor(targets))
    weights = [1.0 / class_counts[t].item() for t in targets]
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


def prepare_loaders(
    mod_path: Path,
    train_transform: transforms.Compose,
    val_transform: transforms.Compose,
    batch_size: int = 64,
    generator: Optional[torch.Generator] = None,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    train_ds = datasets.ImageFolder(mod_path / "train", transform=train_transform)
    val_ds = datasets.ImageFolder(mod_path / "val", transform=val_transform)

    sampler = make_weighted_sampler(train_ds.targets, generator=generator)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return train_loader, val_loader, train_ds.classes


class ClassAwareImageFolder(datasets.ImageFolder):
    """
    ImageFolder с per-class аугментацией.

    transform_fn : callable(class_name: str, is_train: bool) -> transforms.Compose
    class_aware  : если False, класс передаётся, но transform_fn вправе его игнорировать
    """

    def __init__(
        self,
        root: Path,
        transform_fn: Callable[[str, bool], transforms.Compose],
        class_aware: bool = True,
        is_train: bool = True,
        **kwargs,
    ):
        super().__init__(root, **kwargs)
        self.transform_fn = transform_fn
        self.class_aware = class_aware
        self.is_train = is_train

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        class_name = self.classes[target]
        tfm = self.transform_fn(class_name, self.is_train and self.class_aware)
        sample = tfm(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target
