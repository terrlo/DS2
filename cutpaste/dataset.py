"""
This file is adapted from: https://github.com/Runinho/pytorch-cutpaste
"""
import glob
from torch.utils.data import Dataset
from PIL import Image

# from joblib import Parallel, delayed
import random, math
import torch
from torchvision import transforms


class CutPaste(object):
    """Base class for both cutpaste variants with common operations"""

    def __init__(self, colorJitter=0.1, transform=None):
        self.transform = transform

        if colorJitter is None:
            self.colorJitter = None
        else:
            self.colorJitter = transforms.ColorJitter(
                brightness=colorJitter,
                contrast=colorJitter,
                saturation=colorJitter,
                hue=colorJitter,
            )

    def __call__(self, org_img, img):
        # apply transforms to both images
        if self.transform:
            img = self.transform(img)
            org_img = self.transform(org_img)
        return org_img, img


class CutPasteNormal(CutPaste):
    """Randomly copy one patche from the image and paste it somewere else.
    Args:
        area_ratio (list): list with 2 floats for maximum and minimum area to cut out
        aspect_ratio (float): minimum area ration. Ration is sampled between aspect_ratio and 1/aspect_ratio.
    """

    def __init__(self, area_ratio=[0.02, 0.15], aspect_ratio=0.3, **kwags):
        super(CutPasteNormal, self).__init__(**kwags)
        self.area_ratio = area_ratio
        self.aspect_ratio = aspect_ratio

    def __call__(self, img):
        h = img.size[0]
        w = img.size[1]

        # ratio between area_ratio[0] and area_ratio[1]
        ratio_area = random.uniform(self.area_ratio[0], self.area_ratio[1]) * w * h

        # sample in log space
        log_ratio = torch.log(torch.tensor((self.aspect_ratio, 1 / self.aspect_ratio)))
        aspect = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

        cut_w = int(round(math.sqrt(ratio_area * aspect)))
        cut_h = int(round(math.sqrt(ratio_area / aspect)))

        from_location_h = int(random.uniform(0, h - cut_h))
        from_location_w = int(random.uniform(0, w - cut_w))

        box = [
            from_location_w,
            from_location_h,
            from_location_w + cut_w,
            from_location_h + cut_h,
        ]
        patch = img.crop(box)

        if self.colorJitter:
            patch = self.colorJitter(patch)

        to_location_h = int(random.uniform(0, h - cut_h))
        to_location_w = int(random.uniform(0, w - cut_w))

        insert_box = [
            to_location_w,
            to_location_h,
            to_location_w + cut_w,
            to_location_h + cut_h,
        ]
        augmented = img.copy()
        augmented.paste(patch, insert_box)

        return super().__call__(img, augmented)


class CutPasteScar(CutPaste):
    """Randomly copy one patche from the image and paste it somewere else.
    Args:
        width (list): width to sample from. List of [min, max]
        height (list): height to sample from. List of [min, max]
        rotation (list): rotation to sample from. List of [min, max]
    """

    def __init__(self, width=[2, 16], height=[10, 25], rotation=[-45, 45], **kwags):
        super(CutPasteScar, self).__init__(**kwags)
        self.reduction_raio = (
            64 / 256
        )  # account for the reduction in patch size: image_size=256, patch_size=64
        self.width = [
            max(1, round(item * self.reduction_raio + 0.01)) for item in width
        ]  # [1,4]
        self.height = [
            round(height[idx] / width[idx] * item + 0.01)
            for idx, item in enumerate(self.width)
        ]  # [5,6]
        self.rotation = rotation

    def __call__(self, img):
        h = img.size[0]
        w = img.size[1]

        # cut region
        cut_w = random.uniform(*self.width)
        cut_h = random.uniform(*self.height)

        from_location_h = int(random.uniform(0, h - cut_h))
        from_location_w = int(random.uniform(0, w - cut_w))

        box = [
            from_location_w,
            from_location_h,
            from_location_w + cut_w,
            from_location_h + cut_h,
        ]
        patch = img.crop(box)

        if self.colorJitter:
            patch = self.colorJitter(patch)

        # rotate
        rot_deg = random.uniform(*self.rotation)
        patch = patch.convert("RGBA").rotate(rot_deg, expand=True)

        # paste
        to_location_h = int(random.uniform(0, h - patch.size[0]))
        to_location_w = int(random.uniform(0, w - patch.size[1]))

        mask = patch.split()[-1]
        patch = patch.convert("RGB")

        augmented = img.copy()
        augmented.paste(patch, (to_location_w, to_location_h), mask=mask)

        return super().__call__(img, augmented)


class CutPaste3Way(object):
    def __init__(self, **kwags):
        self.normal = CutPasteNormal(**kwags)
        self.scar = CutPasteScar(**kwags)

    def __call__(self, img):
        # the input img has already gone through ColorJitter and RandomCrop
        org, cutpaste_normal = self.normal(img)
        _, cutpaste_scar = self.scar(img)

        return org, cutpaste_normal, cutpaste_scar


class Repeat(Dataset):
    def __init__(self, org_dataset, new_length):
        # new_length is just a dummy input for len output, with no actual effect
        self.org_dataset = org_dataset
        self.org_length = len(self.org_dataset)
        self.new_length = new_length

    def __len__(self):
        return self.new_length

    def __getitem__(self, idx):
        return self.org_dataset[
            idx % self.org_length
        ]  # if idx exceeds original dataset length, then do idx % self.org_length


class MVTecTrainCutPaste(Dataset):
    def __init__(self, image_size=256, category="all", transform=None):
        self.transform = transform
        self.image_size = image_size
        self.image_names = (
            glob.glob("./dataset/mvtec_train/*/*.png")
            if category == "all"
            else glob.glob(f"./dataset/mvtec_train/{category}/*.png")
        )

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img = (
            Image.open(self.image_names[idx])
            .resize((self.image_size, self.image_size))
            .convert("RGB")
        )
        if self.transform is not None:
            img = self.transform(img)
        return img
