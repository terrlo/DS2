import os
import random
import math
import torch.utils.data as data
from PIL import Image, ImageOps
from contrast.data.transform import GaussianBlur
from . import transform_coord
import torchvision.transforms as T


normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir):
    """Find all folder names within `dir`
    Args:
        dir (string): parent directory
    Returns:
        classes: str[], sorted
        class_to_idx: dict[str, idx]
    """
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(
    dir,
    total_num_classes,
    class_to_idx,
    extensions,
    dataset_portion=1.0,
    instance_loss_weight=0.0,
    instance_loss_func="Cosine",
    seed=2,
):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        # examples: target: zipper
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            random.seed(seed)
            random.shuffle(fnames)
            fnames = fnames[: math.floor(len(fnames) * dataset_portion)]
            for fname in fnames:
                if has_file_allowed_extension(fname, extensions):
                    # examples: target: zipper; fname: 200.png
                    path = os.path.join(root, fname)

                    if instance_loss_weight > 0.0 and instance_loss_func in [
                        "DistAug_MOCOv2",
                        "DistAug_SimCLR",
                    ]:
                        for deg90multiplier in [0, 1, 2, 3]:
                            # total class idx to be quadrupled (each rotation is treated as a new class), also include rotation degree
                            images.append(
                                (
                                    path,
                                    class_to_idx[target]
                                    + total_num_classes * deg90multiplier,
                                    deg90multiplier,
                                )
                            )
                    elif instance_loss_weight > 0.0 and instance_loss_func == "RotPred":
                        deg1 = random.randint(0, 3)
                        deg2 = random.randint(0, 3)
                        images.append(
                            (
                                path,
                                class_to_idx[target],
                                (deg1, deg2),
                            )
                        )
                    else:
                        item = (path, class_to_idx[target])
                        images.append(item)

    return images


class DatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::
        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext
        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext
    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
     Attributes:
        samples (list): List of (sample path, class_index) tuples
    """

    def __init__(
        self,
        root,
        loader,
        extensions,
        transform=None,
        target_transform=None,
        dataset="MVTecAD",
        aug_type=None,
        dense_loss_func=None,
        crop=0.08,
        image_size=224,
        dataset_portion=1.0,
        seed=2,
        instance_loss_weight=0.0,
        instance_loss_func="Cosine",
        instance_branch_class_aware=False,
    ):
        if dataset == "MVTecAD":
            classes, class_to_idx = find_classes(root)
            samples = make_dataset(
                root,
                len(classes),
                class_to_idx,
                extensions,
                dataset_portion=dataset_portion,
                instance_loss_func=instance_loss_func,
                instance_loss_weight=instance_loss_weight,
                seed=seed,
            )
        else:
            raise Exception("Unsupported dataset is specified")

        if len(samples) == 0:
            raise (
                RuntimeError(
                    "Found 0 files in subfolders of: " + root + "\n"
                    "Supported extensions are: " + ",".join(extensions)
                )
            )

        self.root = root
        self.loader = loader
        self.extensions = extensions
        self.samples = samples
        self.labels = (
            [y_1k for _, y_1k, _ in samples]
            if (
                instance_loss_weight > 0.0
                and instance_loss_func
                in ["DistAug_MOCOv2", "DistAug_SimCLR", "RotPred"]
            )
            else [y_1k for _, y_1k in samples]
        )
        self.classes = list(set(self.labels))
        self.transform = transform
        self.target_transform = target_transform
        self.aug_type = aug_type
        self.dense_loss_func = dense_loss_func
        self.image_size = image_size
        self.dataset_portion = dataset_portion
        self.instance_loss_func = instance_loss_func
        self.instance_branch_class_aware = instance_branch_class_aware
        self.instance_loss_weight = instance_loss_weight
        self.crop = crop

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        # ignored
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        fmt_str += "    Root Location: {}\n".format(self.root)
        tmp = "    Transforms (if any): "
        fmt_str += "{0}{1}\n".format(
            tmp, self.transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        tmp = "    Target Transforms (if any): "
        fmt_str += "{0}{1}".format(
            tmp, self.target_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        return fmt_str


IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif"]


def pil_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        img.load()
        return img.convert("RGB")


def default_img_loader(path):
    return pil_loader(path)


class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way:
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        loader=default_img_loader,
        dataset="MVTecAD",
        aug_type=None,
        dense_loss_func=None,
        crop=0.08,
        image_size=224,
        dataset_portion=1.0,
        instance_loss_weight=0.0,
        instance_loss_func="Cosine",
        instance_branch_class_aware=False,
        seed=2,
    ):
        super(ImageFolder, self).__init__(
            root,
            loader,
            IMG_EXTENSIONS,
            transform=transform,
            target_transform=target_transform,
            dataset=dataset,
            aug_type=aug_type,
            dense_loss_func=dense_loss_func,
            image_size=image_size,
            dataset_portion=dataset_portion,
            seed=seed,
            instance_loss_weight=instance_loss_weight,
            instance_loss_func=instance_loss_func,
            crop=crop,
            instance_branch_class_aware=instance_branch_class_aware,
        )

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        if self.instance_loss_weight > 0.0 and self.instance_loss_func in [
            "DistAug_MOCOv2",
            "DistAug_SimCLR",
        ]:
            # use rotation for distribution augmentation
            path, target, deg90multiplier = self.samples[index]
            if isinstance(path, str):
                original_image = self.loader(path)
            elif isinstance(path, Image.Image):
                original_image = path

            image = original_image.rotate(90 * deg90multiplier)
        elif self.instance_loss_weight > 0.0 and self.instance_loss_func == "RotPred":
            # load image as usual, but also include the additional "deg90multiplier" information
            path, target, (deg90multiplier1, deg90multiplier2) = self.samples[index]
            if isinstance(path, str):
                image = self.loader(path)
            elif isinstance(path, Image.Image):
                image = path
        else:
            path, target = self.samples[index]
            if isinstance(path, str):
                image = self.loader(path)
            elif isinstance(path, Image.Image):
                image = path

        if self.aug_type in ["BYOL", "MVTec_AUG", "DistAug"]:
            # Rotation Prediction
            if self.instance_loss_weight > 0.0 and self.instance_loss_func == "RotPred":
                # RotPred has its own data augmentation pipeline, not using self.transform

                # get views
                view1, coord1 = transform_coord.Compose(
                    [
                        transform_coord.RandomResizedCropCoord(
                            self.image_size, scale=(self.crop, 1.0)
                        ),
                        transform_coord.RandomHorizontalFlipCoord(),
                    ]
                )(image)

                view2, coord2 = transform_coord.Compose(
                    [
                        transform_coord.RandomResizedCropCoord(
                            self.image_size, scale=(self.crop, 1.0)
                        ),
                        transform_coord.RandomHorizontalFlipCoord(),
                    ]
                )(image)

                # get rotate views
                view1_rot = view1.rotate(90 * deg90multiplier1)
                view2_rot = view2.rotate(90 * deg90multiplier2)

                # tensorize these views (plug some augmentations)
                view1_tensor, _ = transform_coord.Compose(
                    [
                        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                        T.RandomApply([GaussianBlur()], p=1.0),
                        T.ToTensor(),
                        normalize,
                    ]
                )(view1)
                view2_tensor, _ = transform_coord.Compose(
                    [
                        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                        T.RandomApply([GaussianBlur()], p=0.1),
                        T.ToTensor(),
                        normalize,
                    ]
                )(view2)
                view1_rot_tensor, _ = transform_coord.Compose(
                    [
                        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                        T.RandomApply([GaussianBlur()], p=0.5),
                        T.RandomGrayscale(p=0.2),
                        T.RandomApply([ImageOps.solarize], p=0.3),
                        T.ToTensor(),
                        normalize,
                    ]
                )(view1_rot)
                view2_rot_tensor, _ = transform_coord.Compose(
                    [
                        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                        T.RandomApply([GaussianBlur()], p=0.5),
                        T.RandomGrayscale(p=0.2),
                        T.RandomApply([ImageOps.solarize], p=0.3),
                        T.ToTensor(),
                        normalize,
                    ]
                )(view2_rot)

                return (
                    view1_tensor,
                    view2_tensor,
                    coord1,
                    coord2,
                    view1_rot_tensor,
                    view2_rot_tensor,
                    deg90multiplier1,
                    deg90multiplier2,
                )

            # default code
            img1 = self.transform[0](image)
            img2 = self.transform[1](image)
            img1, coord1 = img1
            img2, coord2 = img2
            if self.instance_branch_class_aware:
                return img1, img2, coord1, coord2, target
            else:
                return img1, img2, coord1, coord2

        elif self.aug_type in ["SimCLR", "MOCOv2"]:
            img1 = self.transform(image)
            img2 = self.transform(image)
            img1, coord1 = img1
            img2, coord2 = img2

            return img1, img2, coord1, coord2, target
