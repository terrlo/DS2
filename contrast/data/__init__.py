from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .transform import get_transform
from .dataset import ImageFolder


def get_loader(args):
    # decide augmentation method
    if args.instance_loss_weight == 0.0:
        aug_type = "MVTec_AUG"
    else:
        if args.instance_loss_func == "Cosine":
            aug_type = "BYOL"
        elif args.instance_loss_func in ["DistAug_MOCOv2", "DistAug_SimCLR", "RotPred"]:
            # RotPred uses its own augmentation, here DistAug is just a dummy
            aug_type = "DistAug"
        elif args.instance_loss_func == "MOCOv2":
            aug_type = "MOCOv2"
        elif args.instance_loss_func == "SimCLR":
            aug_type = "SimCLR"
        else:
            raise ValueError(
                f"Invalid instance_loss_func option: {args.instance_loss_func}"
            )

    transform = get_transform(aug_type, args.crop, args.image_size)

    train_dataset = ImageFolder(
        args.data_dir,
        transform=transform,
        aug_type=aug_type,
        crop=args.crop,
        image_size=args.image_size,
        dataset=args.dataset,
        dataset_portion=args.dataset_portion,
        instance_loss_weight=args.instance_loss_weight,
        instance_loss_func=args.instance_loss_func,
        dense_loss_func=args.dense_loss_func,
        instance_branch_class_aware=args.instance_branch_class_aware,
        seed=args.seed,
    )

    # dataloader
    return DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=DistributedSampler(train_dataset),
        drop_last=True,
    )
