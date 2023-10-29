import numpy as np
from PIL import ImageFilter, ImageOps
from torchvision import transforms
from . import transform_coord


class GaussianBlur(object):
    def __call__(self, x):
        sigma = np.random.uniform(0.1, 2.0)
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def get_transform(aug_type, crop, image_size=224):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    if aug_type == "MOCOv2":  # used in MOCOv2
        transform = transform_coord.Compose(
            [
                transform_coord.RandomResizedCropCoord(image_size, scale=(crop, 1.0)),
                transform_coord.RandomHorizontalFlipCoord(),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur()], p=0.5),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif aug_type == "SimCLR":  # used in SimCLR
        transform = transform_coord.Compose(
            [
                transform_coord.RandomResizedCropCoord(image_size, scale=(crop, 1.0)),
                transform_coord.RandomHorizontalFlipCoord(),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur()], p=0.5),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif aug_type == "BYOL":
        transform_1 = transform_coord.Compose(
            [
                transform_coord.RandomResizedCropCoord(image_size, scale=(crop, 1.0)),
                transform_coord.RandomHorizontalFlipCoord(),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8
                ),
                transforms.RandomApply([GaussianBlur()], p=1.0),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                normalize,
            ]
        )
        transform_2 = transform_coord.Compose(
            [
                transform_coord.RandomResizedCropCoord(image_size, scale=(crop, 1.0)),
                transform_coord.RandomHorizontalFlipCoord(),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8
                ),
                transforms.RandomApply([GaussianBlur()], p=0.1),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([ImageOps.solarize], p=0.5),
                transforms.ToTensor(),
                normalize,
            ]
        )
        transform = (transform_1, transform_2)
    elif aug_type == "DistAug":
        # augmentation for baseline paper
        transform_1 = transform_coord.Compose(
            [
                transform_coord.RandomResizedCropCoord(image_size, scale=(0.5, 1.0)),
                transform_coord.RandomHorizontalFlipCoord(),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)], p=0.8
                ),
                transforms.RandomApply([GaussianBlur()], p=0.5),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                normalize,
            ]
        )
        transform_2 = transform_coord.Compose(
            [
                transform_coord.RandomResizedCropCoord(image_size, scale=(0.5, 1.0)),
                transform_coord.RandomHorizontalFlipCoord(),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)], p=0.8
                ),
                transforms.RandomApply([GaussianBlur()], p=0.5),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                normalize,
            ]
        )

        transform = (transform_1, transform_2)
    elif aug_type == "MVTec_AUG":
        transform_1 = transform_coord.Compose(
            [
                transform_coord.RandomResizedCropCoord(image_size, scale=(crop, 1.0)),
                transform_coord.RandomHorizontalFlipCoord(),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8
                ),
                transforms.RandomApply([GaussianBlur()], p=1.0),
                transforms.ToTensor(),
                normalize,
            ]
        )
        transform_2 = transform_coord.Compose(
            [
                transform_coord.RandomResizedCropCoord(image_size, scale=(crop, 1.0)),
                transform_coord.RandomHorizontalFlipCoord(),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8
                ),
                transforms.RandomApply([GaussianBlur()], p=0.1),
                transforms.ToTensor(),
                normalize,
            ]
        )
        transform = (transform_1, transform_2)
    else:
        supported = "[MOCOv2, SimCLR, BYOL, MVTec_AUG, DistAug]"
        raise NotImplementedError(
            f'aug_type "{aug_type}" not supported. Should in {supported}'
        )

    return transform
