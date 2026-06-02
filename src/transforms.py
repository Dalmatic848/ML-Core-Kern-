from torchvision import transforms


class PadToSquare:
    """Дополняет изображение серым цветом до квадрата без изменения соотношения сторон."""

    def __init__(self, fill: int = 128):
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        if w == h:
            return img
        pad = abs(w - h) // 2
        padding = (0, pad, 0, pad) if w > h else (pad, 0, pad, 0)
        return transforms.Pad(padding, fill=self.fill)(img)


def get_transforms(
    resize_strategy: str = "pad",
    aug_level: str = "std",
    is_train: bool = True,
) -> transforms.Compose:
    """
    resize_strategy : 'pad'  — PadToSquare → Resize(224)
                      'crop' — RandomResizedCrop (train) / CenterCrop (val)
    aug_level       : 'heavy' | 'std' | 'light' | 'none'

    Порядок: сначала resize → потом аугментация → ToTensor → Normalize.
    Такой порядок гарантирует, что пространственные параметры aug (translate,
    rotate) применяются к фиксированному масштабу 224×224.
    """
    ops = []

    # 1. Resize
    if resize_strategy == "pad":
        ops.append(PadToSquare(fill=128))
        ops.append(transforms.Resize((224, 224)))
    elif resize_strategy == "crop":
        if is_train:
            ops.append(transforms.RandomResizedCrop(224, scale=(0.85, 1.0)))
        else:
            ops.append(transforms.Resize(256))
            ops.append(transforms.CenterCrop(224))

    # 2. Аугментация (только train)
    if is_train:
        if aug_level == "heavy":
            ops.append(transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2))
            ops.append(transforms.RandomRotation(15))
            ops.append(transforms.RandomHorizontalFlip(p=0.5))
            ops.append(transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)))
        elif aug_level == "std":
            ops.append(transforms.ColorJitter(brightness=0.2, contrast=0.2))
            ops.append(transforms.RandomHorizontalFlip(p=0.5))
            ops.append(transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)))
        elif aug_level == "light":
            ops.append(transforms.RandomHorizontalFlip(p=0.5))
        # aug_level == 'none' → без аугментации

    # 3. Нормализация
    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    return transforms.Compose(ops)
