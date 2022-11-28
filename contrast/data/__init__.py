from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .transform import get_transform
from .dataset import ImageFolder


def get_loader(args):
    transform = get_transform(args.aug, args.crop, args.image_size)

    train_dataset = ImageFolder(
        args.data_dir,
        transform=transform,
        aug_type=args.aug,
        crop=args.crop,
        image_size=args.image_size,
        dataset=args.dataset,
        dataset_portion=args.dataset_portion,
        instance_loss_weight=args.instance_loss_weight,
        instance_loss_func=args.instance_loss_func,
        dense_loss_func=args.dense_loss_func,
        dense_branch_class_aware=args.dense_branch_class_aware,
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
