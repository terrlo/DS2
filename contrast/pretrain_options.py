import argparse
from datetime import datetime
from contrast import resnet
from contrast.util import MyHelpFormatter

model_names = sorted(
    name
    for name in resnet.__all__
    if name.islower() and callable(resnet.__dict__[name])
)


def parse_option():
    parser = argparse.ArgumentParser(
        f"contrast pretraining stage", formatter_class=MyHelpFormatter
    )

    # dataset
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./dataset/mvtec_train/",
        help="dataset directory",
    )
    parser.add_argument(
        "--crop",
        type=float,
        default=0.08,
        help="minimum crop",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MVTecAD",
        choices=[
            "MVTecAD",
        ],
        help="dataset name",
    )
    parser.add_argument(
        "--dataset_portion",
        type=float,
        default=1.0,
        help="percentage of dataset images fed into the network",
    )
    parser.add_argument("--image-size", type=int, default=224, help="image crop size")
    parser.add_argument(
        "--in_channel", type=int, default=3, help="input color channels"
    )

    # model
    parser.add_argument(
        "--arch",
        type=str,
        default="resnet18",
        choices=model_names,
        help="backbone architecture",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        default="DS2",
        choices=["DS2", "CutPaste"],
        help="architecture choice",
    )
    parser.add_argument(
        "--cutpaste_category",
        type=str,
        required=False,
        default="all",
        choices=[
            "all",
            "carpet",
            "grid",
            "leather",
            "tile",
            "wood",
            "bottle",
            "cable",
            "capsule",
            "hazelnut",
            "metal_nut",
            "pill",
            "screw",
            "toothbrush",
            "transistor",
            "zipper",
        ],
        help="pretrain category for cutpaste",
    )
    parser.add_argument(
        "--feature-dim", type=int, default=256, help="feature dimension"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="num of workers per GPU to use"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="batch_size for single gpu"
    )
    parser.add_argument(
        "--head_dims",
        action="store",
        type=str,
        required=False,
        default="512,512,512,512,512,512,512,512,128",
        help="head projector's dimensions for baseline paper",
    )

    # optimization
    parser.add_argument(
        "--base-learning-rate",
        "--base-lr",
        type=float,
        default=0.03,
        help="base learning rate",
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["sgd", "lars"],
        default="sgd",
        help="optimizer choice",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="cosine",
        choices=["step", "cosine"],
        help="learning rate scheduler",
    )
    parser.add_argument("--warmup-epoch", type=int, default=5, help="warmup epoch")
    parser.add_argument(
        "--warmup-multiplier", type=int, default=100, help="warmup multiplier"
    )
    parser.add_argument(
        "--lr-decay-epochs",
        type=int,
        default=[120, 160, 200],
        nargs="+",
        help="for step scheduler. where to decay lr, can be a list",
    )
    parser.add_argument(
        "--lr-decay-rate",
        type=float,
        default=0.1,
        help="for step scheduler. decay rate for learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="weight decay",
    )
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum for SGD")
    parser.add_argument("--start-epoch", type=int, default=1, help="start epoch")
    parser.add_argument(
        "--epochs", type=int, default=100, help="number of training epochs"
    )

    # PixPro arguments
    parser.add_argument("--pixpro-p", type=float, default=2.0)
    parser.add_argument(
        "--pixpro-momentum",
        type=float,
        default=0.99,
        help="momentume parameter used in MoCo and InstDisc",
    )
    parser.add_argument("--pixpro-clamp-value", type=float, default=0.0)
    parser.add_argument("--pixpro-transform-layer", type=int, default=1)

    # loss functions
    parser.add_argument(
        "--instance_loss_weight",
        type=float,
        default=0.0,
        help="instance branch loss weight",
    )
    parser.add_argument(
        "--dense_loss_weight",
        type=float,
        default=1.0,
        help="dense branch loss weight",
    )
    parser.add_argument(
        "--instance_loss_func",
        type=str,
        default="Cosine",
        choices=[
            "Cosine",
            "DistAug_MOCOv2",
            "DistAug_SimCLR",
            "RotPred",
            "MOCOv2",
            "SimCLR",
        ],
        help="loss function for instance branch.",
    )
    parser.add_argument(
        "--dense_loss_func",
        type=str,
        default="DS2",
        choices=["DS2"],
        help="loss function for dense branch",
    )

    # misc
    parser.add_argument(
        "--output-dir", type=str, default="./output", help="output director"
    )
    parser.add_argument("--print-freq", type=int, default=20, help="print frequency")
    parser.add_argument("--save-freq", type=int, default=10, help="save frequency")
    parser.add_argument(
        "--local_rank",
        type=int,
        required=True,
        help="local rank for DistributedDataParallel",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=datetime.now().strftime("%Y%m%d-%H%M%S"),
        help="timestamp for current pre-training",
    )
    parser.add_argument(
        "--debug_mode",
        action="store_true",
        help="debug mode",
    )
    parser.add_argument(
        "--note",
        action="store",
        type=str,
        required=False,
        help="any personal note for this pretraining",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="random seed",
    )

    args = parser.parse_args()

    return args
