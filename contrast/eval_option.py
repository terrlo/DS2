import argparse


def parse_option():
    parser = argparse.ArgumentParser()

    # directories
    parser.add_argument(
        "--pretrained_model_dir",
        action="store",
        type=str,
        help="",
    )
    parser.add_argument(
        "--mvtec_dataset_dir",
        action="store",
        type=str,
        default="./dataset/mvtec/",
        help="",
    )
    parser.add_argument(
        "--dataset",
        action="store",
        type=str,
        choices=["MVTEC", "KSDD2", "MTD", "LOCO"],
        default="MVTEC",
    )
    parser.add_argument(
        "--log_dir",
        action="store",
        type=str,
        default="./logs/",
        help="",
    )
    parser.add_argument(
        "--qualitative_dir",
        action="store",
        type=str,
        help="",
        default="./qualitative/",
    )

    #  mvtec
    parser.add_argument(
        "--category",
        action="store",
        type=str,
        required=False,
        default="all",
        help="MVTec category type",
    )
    parser.add_argument(
        "--qualitative",
        action="store_true",
        required=False,
        help="print qualitative images",
    )

    # size
    parser.add_argument(
        "--patch_size", type=int, default=32, help="patch image side length"
    )
    parser.add_argument(
        "--patch_stride", type=int, default=4, help="patch cutting stride"
    )
    parser.add_argument(
        "--gde_smp_size",
        type=int,
        default=350,
        help="number of images to estimate KDE",
    )
    parser.add_argument(
        "--resized_image_size", type=int, default=256, help="resized image resolution"
    )

    # misc
    parser.add_argument(
        "--note",
        action="store",
        type=str,
        required=False,
        help="any personal note for this evaluation",
    )
    parser.add_argument(
        "--ckpt_epoch",
        nargs="+",
        required=False,
        help="which epoch checkpoint(s) to use for evaluation",
    )
    parser.add_argument("--model", type=str, choices=["DS2", "CutPaste"], default="DS2")
    parser.add_argument(
        "--gde_count", type=str, choices=["one", "patchwise"], default="patchwise"
    )

    parsed_args = parser.parse_args()

    return parsed_args
