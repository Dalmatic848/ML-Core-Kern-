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
    num_workers: int = 4,
    pin_memory: bool = True,
    use_sampler: bool = True,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    train_ds = datasets.ImageFolder(mod_path / "train", transform=train_transform)
    val_ds = datasets.ImageFolder(mod_path / "val", transform=val_transform)

    if use_sampler:
        sampler = make_weighted_sampler(train_ds.targets, generator=generator)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            persistent_workers=num_workers > 0,
            generator=generator,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            persistent_workers=num_workers > 0,
            generator=generator,
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader, train_ds.classes


class PairedDataset(torch.utils.data.Dataset):
    """Синхронная загрузка ДС и УФ тайлов одного физического фрагмента.

    Структура: dataset/{ДС|УФ}/{split}/{class}/*.jpg
    Имена файлов идентичны между модальностями — строим пересечение per-class.
    Возвращает: ((img_ds, img_uv), label)
    """

    def __init__(
        self,
        ds_split_root: Path,
        uv_split_root: Path,
        ds_transform,
        uv_transform,
    ) -> None:
        self.ds_transform = ds_transform
        self.uv_transform = uv_transform

        ds_cls = sorted(d.name for d in ds_split_root.iterdir() if d.is_dir())
        uv_cls = sorted(d.name for d in uv_split_root.iterdir() if d.is_dir())
        self.classes = sorted(set(ds_cls) & set(uv_cls))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        samples: list = []
        for cls in self.classes:
            ds_files = {f.name: f for f in (ds_split_root / cls).glob('*.jpg')}
            uv_files = {f.name: f for f in (uv_split_root / cls).glob('*.jpg')}
            label = self.class_to_idx[cls]
            for name in sorted(set(ds_files) & set(uv_files)):
                samples.append((ds_files[name], uv_files[name], label))
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        ds_path, uv_path, label = self.samples[idx]
        from PIL import Image as PILImage
        img_ds = PILImage.open(ds_path).convert('RGB')
        img_uv = PILImage.open(uv_path).convert('RGB')
        return (self.ds_transform(img_ds), self.uv_transform(img_uv)), label

    @property
    def targets(self) -> List[int]:
        return [s[2] for s in self.samples]


class SoftLabelPairedDataset(torch.utils.data.Dataset):
    """PairedDataset с мягкими метками (float вектор вместо int класса).

    Читает soft_labels.json из dataset_root — маппинг {class_name: [p0, p1, ...]}
    и base_classes.json — список базовых компонент.
    Структура: dataset/{ДС|УФ}/{split}/{orig_class}/*.jpg
    """

    def __init__(
        self,
        ds_split_root: Path,
        uv_split_root: Path,
        ds_transform,
        uv_transform,
        soft_labels: dict,
    ) -> None:
        self.ds_transform = ds_transform
        self.uv_transform = uv_transform
        self.soft_labels = soft_labels

        ds_cls = sorted(d.name for d in ds_split_root.iterdir() if d.is_dir())
        uv_cls = sorted(d.name for d in uv_split_root.iterdir() if d.is_dir())
        classes = sorted(set(ds_cls) & set(uv_cls) & set(soft_labels.keys()))
        self.classes = classes

        samples: list = []
        for cls in classes:
            ds_files = {f.name: f for f in (ds_split_root / cls).glob('*.jpg')}
            uv_files = {f.name: f for f in (uv_split_root / cls).glob('*.jpg')}
            soft_vec = torch.tensor(soft_labels[cls], dtype=torch.float32)
            for name in sorted(set(ds_files) & set(uv_files)):
                samples.append((ds_files[name], uv_files[name], soft_vec))
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        ds_path, uv_path, soft_label = self.samples[idx]
        from PIL import Image as PILImage
        img_ds = PILImage.open(ds_path).convert('RGB')
        img_uv = PILImage.open(uv_path).convert('RGB')
        return (self.ds_transform(img_ds), self.uv_transform(img_uv)), soft_label

    @property
    def targets(self) -> List[int]:
        return [int(s[2].argmax().item()) for s in self.samples]


def prepare_paired_loaders(
    dataset_root: Path,
    ds_train_tfm,
    ds_val_tfm,
    uv_train_tfm,
    uv_val_tfm,
    batch_size: int = 64,
    generator: Optional[torch.Generator] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    train_ds = PairedDataset(
        dataset_root / 'ДС' / 'train',
        dataset_root / 'УФ' / 'train',
        ds_train_tfm, uv_train_tfm,
    )
    val_ds = PairedDataset(
        dataset_root / 'ДС' / 'val',
        dataset_root / 'УФ' / 'val',
        ds_val_tfm, uv_val_tfm,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        drop_last=True, persistent_workers=num_workers > 0,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
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
