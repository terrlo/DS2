import argparse


def parse_option():
    parser = argparse.ArgumentParser()

    # directories
    parser.add_argument(
        "--pretrained_model_dir",
        action="store",
        type=str,
        required=True,
        help="",
    )
    parser.add_argument(
        "--mvtec_dataset_dir",
        action="store",
        type=str,
        required=True,
        help="",
    )
    parser.add_argument(
        "--log_dir",
        action="store",
        type=str,
        required=True,
        help="",
    )
    parser.add_argument(
        "--qualitative_dir",
        action="store",
        type=str,
        required=True,
        help="",
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
        default=250,
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
        "--head_type", type=str, default="early_return", help="choose head type"
    )
    parser.add_argument(
        "--ckpt_epoch",
        nargs="+",
        required=True,
        help="which epoch checkpoint(s) to use for evaluation",
    )

    parsed_args = parser.parse_args()

    return parsed_args
