from argparse import Namespace
import os
import torch
from contrast.logger import setup_logger
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import torch.nn as nn
from contrast import resnet
from contrast.density import GaussianDensityTorch
from contrast.eval_option import parse_option
from contrast.mvtec_dataloader import MVTecTestDataset, MVTecTrainDataset
from contrast.loco_dataloader import LocoTestDataset, LocoTrainDataset
from datetime import datetime
from scipy import signal
import cv2
from sklearn.decomposition import PCA


device = "cuda" if torch.cuda.is_available() else "cpu"

# mvtec dataset categories
mvtec_category_list = [
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
]

texture_types = [
    "carpet",
    "grid",
    "leather",
    "tile",
    "wood",
]

object_types = [
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
]

loco_category_list = [
    "breakfast_box",
    "juice_bottle",
    "pushpins",
    "screw_bag",
    "splicing_connectors",
]


""" 
********************************
Gaussian Smoothing (Upsampling)
********************************
"""


def normalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def kernel_size_to_std(k: int):
    """Returns a standard deviation value for a Gaussian kernel based on its size"""
    return np.log10(0.45 * k + 1) + 0.25 if k < 32 else 10


def gkern(k: int):
    "" "Returns a 2D Gaussian kernel array with given kernel size k and std std " ""
    std = kernel_size_to_std(k)
    if k % 2 == 0:
        # if kernel size is even, signal.gaussian returns center values sampled from gaussian at x=-1 and x=1
        # which is much less than 1.0 (depending on std). Instead, sample with kernel size k-1 and duplicate center
        # value, which is 1.0. Then divide whole signal by 2, because the duplicate results in a too high signal.
        gkern1d = signal.gaussian(k - 1, std=std).reshape(k - 1, 1)
        gkern1d = np.insert(gkern1d, (k - 1) // 2, gkern1d[(k - 1) // 2]) / 2
    else:
        gkern1d = signal.gaussian(k, std=std).reshape(k, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d


def receptive_upsample(pixels: torch.Tensor, args: Namespace) -> torch.Tensor:
    """
    Implement this to upsample given tensor images based on the receptive field with a Gaussian kernel.
    Usually one can just invoke the receptive_upsample method of the last convolutional layer.
    :param pixels: tensors that are to be upsampled (n x c x h x w)
    """
    assert (
        pixels.dim() == 4 and pixels.size(1) == 1
    ), "receptive upsample works atm only for one channel"
    gaus = torch.from_numpy(gkern(args.patch_size)).float().to(pixels.device)

    res = torch.nn.functional.conv_transpose2d(
        pixels,
        gaus.unsqueeze(0).unsqueeze(0),
        stride=args.patch_stride,
    )

    return res.to(device)


""" 
********************************
Evaluation
********************************
"""


def eval_on_device(categories, args: Namespace):
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    logger.info(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
    )
    logger.info("*************************")

    # image_level_auroc_all_categories = []
    # image_level_auroc_texture_categories = []
    # image_level_auroc_object_categories = []

    pixel_level_auroc_all_categories = []
    pixel_level_auroc_texture_categories = []
    pixel_level_auroc_object_categories = []

    """
    ********************************
    Load Checkpoint
    ********************************
    """
    checkpoint = torch.load(
        os.path.join(
            args.pretrained_model_dir,
            f"cutpaste.pth"
            if args.model == "CutPaste"
            else f"ckpt_epoch_{args.ckpt_epoch}.pth",
        ),
        map_location=device,
    )
    checkpoint_obj = checkpoint["model"]
    pretrained_model = {}
    target_module = (
        "module.resnet18." if args.model == "CutPaste" else "module.encoder."
    )
    for k, v in checkpoint_obj.items():
        if not k.startswith(target_module):
            continue
        k = k.replace(target_module, "")
        pretrained_model[k] = v

    encoder = resnet.__dict__["resnet18"]
    encoder.load_state_dict(pretrained_model)
    encoder = encoder.to(device)
    encoder.eval()  # set model to eval mode

    for category in categories:
        if args.qualitative:
            image_out_path = (
                f"./qualitative_{timestamp}/{category}"  # image output directory
            )
            if not os.path.exists(image_out_path):
                os.makedirs(image_out_path)

        """
        ********************************
        Create Dataloader for Normal Images
        ********************************
        """

        # get embeddings from training dataset
        if args.dataset == "MVTEC":
            train_dataset = MVTecTrainDataset(
                os.path.join(args.mvtec_dataset_dir, category, "train/good/"),
                resize_shape=[args.resized_image_size, args.resized_image_size],
            )
        elif args.dataset == "LOCO":
            train_dataset = LocoTrainDataset(
                f"./dataset/mvtecloco/{category}/train/good/",
                resize_shape=[args.resized_image_size, args.resized_image_size],
            )

        train_dataloader = DataLoader(
            train_dataset, batch_size=args.gde_smp_size, shuffle=True
        )

        train_patches_by_index_dict = {}  # key is index (i,j)

        if category in nonalign_types:
            all_embeds = []

        """ 
        ********************************
        Fit GDE for Normal Images
        ********************************
        """
        for info_batched in train_dataloader:
            train_image_batch = info_batched["image"].to(device)  # shape: bs*3*x*y

            patches_raw = train_image_batch.unfold(
                2, args.patch_size, args.patch_stride
            ).unfold(
                3, args.patch_size, args.patch_stride
            )  # shape: [bs, 3, crop_row, crop_column, patch_size, patch_size]

            bs, _, num_crop_row, num_crop_col, _, _ = patches_raw.shape

            for i in range(num_crop_row):
                for j in range(num_crop_col):
                    slice = patches_raw[:, :, i, j, :, :]
                    embeds = encoder(slice.to(device)).mean(
                        dim=(-2, -1)
                    )  # shape: (bs, C)
                    embeds = nn.functional.normalize(embeds, dim=1)

                    if category in nonalign_types:
                        all_embeds.append(embeds)
                    else:
                        patch_pca = PCA(
                            n_components=50 if category == "toothbrush" else 200
                        )
                        patch_pca.fit(embeds.cpu())

                        embeds_transformed = torch.tensor(
                            patch_pca.transform(embeds.cpu()), dtype=torch.float32
                        ).to(
                            device
                        )  # shape: (bs, 50?)

                        gde_estimator = GaussianDensityTorch()
                        gde_estimator.fit(embeds_transformed)
                        key = f"{i},{j}"
                        train_patches_by_index_dict[key] = (
                            gde_estimator,
                            patch_pca,
                        )  # update value of train_patches_by_index_dict

            break  # the for-loop only execute once

        if category in nonalign_types:
            embeds = torch.cat(
                all_embeds, dim=0
            )  # shape: (bs*num_crop_row*num_crop_col, C)

            patch_pca = PCA(n_components=500)
            patch_pca.fit(embeds.cpu())
            embeds_transformed = torch.tensor(
                patch_pca.transform(embeds.cpu()), dtype=torch.float32
            ).to(
                device
            )  # shape: (bs, 50?)
            gde_estimator = GaussianDensityTorch()
            gde_estimator.fit(embeds_transformed)
            train_patches_estimation = (gde_estimator, patch_pca)

        """ 
        ********************************
        Create Dataloader for Test Images
        ********************************
        """
        # get test dataset
        if args.dataset == "MVTEC":
            test_dataset = MVTecTestDataset(
                os.path.join(args.mvtec_dataset_dir, category, "test/"),
                resize_shape=[args.resized_image_size, args.resized_image_size],
            )
        elif args.dataset == "LOCO":
            test_dataset = LocoTestDataset(
                f"./dataset/mvtecloco/{category}/test/",
                resize_shape=[args.resized_image_size, args.resized_image_size],
            )

        test_dataloader = DataLoader(
            test_dataset, batch_size=200, shuffle=False, num_workers=0
        )

        # image_level_gt_list = []  # image-level ground-truth anomaly score [0,1]
        # image_level_pred_list = []  # image-level predicted anomaly score
        pixel_level_gt_list = []
        pixel_level_pred_list = []

        """ 
        ********************************
        Evaluate Test Images
        ********************************
        """
        for _, info_batched in enumerate(test_dataloader):
            test_image = info_batched["image"].to(device)  # shape: bs*3*x*y

            patches_raw = test_image.unfold(
                2, args.patch_size, args.patch_stride
            ).unfold(
                3, args.patch_size, args.patch_stride
            )  # shape: [bs, 3, crop_row, crop_column, patch_size, patch_size]

            bs, _, num_crop_row, num_crop_col, _, _ = patches_raw.shape

            # raster scan order (first each cols of a row, then each row)
            indices = [
                (i, j) for i in range(num_crop_row) for j in range(num_crop_col)
            ]  # [(0,0), (0,1), ... , (m,n)]

            features = []

            for img_id in range(bs):
                slice = patches_raw[
                    img_id, :, :, :, :, :
                ]  # slice.shape: (3, crop_row, crop_column, patch_size, patch_size)

                test_embeddings = (
                    encoder(
                        slice.reshape(slice.shape[0], -1, *(slice.shape[3:])).transpose(
                            0, 1
                        )  # shape: (crop_row*crop_column, 3, patch_size, patch_size)
                    )
                    .to(device)
                    .mean(dim=(-2, -1))
                )  # shape: (crop_row*crop_column=3249 , feature_dim) for one image
                features.append(test_embeddings)

            features = torch.stack(features)  # features.shape: (bs, 3249, #feature_dim)

            scores = []  # anomaly scores
            crops_count = features.shape[1]
            for index in range(crops_count):
                # each iteration is at a patch index
                patch_row, patch_col = indices[index]  # tuple (i,j)
                embed = features[
                    :, index, :
                ]  # instance embeds shape: (bs , #feature_dim)
                embed = nn.functional.normalize(embed, dim=1)

                if category in nonalign_types:
                    (train_patch_gde_estimator, patch_pca) = train_patches_estimation
                else:
                    (
                        train_patch_gde_estimator,
                        patch_pca,
                    ) = train_patches_by_index_dict[f"{patch_row},{patch_col}"]

                embed = patch_pca.transform(embed.cpu())

                scores_batch = train_patch_gde_estimator.predict(
                    torch.tensor(embed, dtype=torch.float32), device
                )

                scores.append(scores_batch)  # scores_batch.shape: (#test_images)

            scores = torch.stack(scores)  # scores.shape: (3249, #test_images)

            # # image-level score
            # image_level_pred_list.extend(torch.amax(scores, dim=0).cpu().detach().numpy())
            # image_level_gt_list.extend(
            #     info_batched["has_anomaly"].detach().numpy()
            # )  # image_level_gt.shape(#bs),   is_normal: 1.0 or 0.0

            true_mask = info_batched["mask"].detach().numpy()  # shape: (bs,1,x,y)
            # pixel-level upsampling
            scores_t = scores.transpose(0, 1)
            upsampled_scores = receptive_upsample(
                scores_t.reshape(
                    (scores_t.shape[0], num_crop_row, num_crop_col)
                ).unsqueeze(1),
                args,
            )  # first parameter shape: (bs, 1, num_crop_row, num_crop_col); upsampled_scores.shape: (bs, 1, x, y)

            # pixel-level score
            pixel_level_gt_list.extend(true_mask.flatten())
            pixel_level_pred_list.extend(
                upsampled_scores.cpu().detach().numpy().flatten()
            )

            if args.qualitative:
                # qualitative image output
                file_names = info_batched["file_name"]
                raw_images = info_batched["image"]
                heatmap_alpha = 0.5

                for img_idx in range(bs):
                    gt_mask = np.transpose(
                        np.array(true_mask[img_idx] * 255), (1, 2, 0)
                    )
                    gt_img = np.transpose(
                        np.array(raw_images[img_idx] * 255), (1, 2, 0)
                    )
                    pre_mask = np.transpose(
                        np.uint8(
                            normalizeData(
                                upsampled_scores[img_idx].cpu().detach().numpy()
                            )
                            * 255
                        ),
                        (1, 2, 0),
                    )
                    heatmap = cv2.applyColorMap(pre_mask, cv2.COLORMAP_JET)
                    hmap_overlay_gt_img = heatmap * heatmap_alpha + gt_img * (
                        1.0 - heatmap_alpha
                    )

                    cv2.imwrite(
                        f"./qualitative_{timestamp}/{category}/{file_names[img_idx]}_[0]mask_gt.jpg",
                        gt_mask,
                    )
                    cv2.imwrite(
                        f"./qualitative_{timestamp}/{category}/{file_names[img_idx]}_[1]heatmap.jpg",
                        hmap_overlay_gt_img,
                    )
                    cv2.imwrite(
                        f"./qualitative_{timestamp}/{category}/{file_names[img_idx]}_[2]img_gt.jpg",
                        gt_img,
                    )

        """ 
        ********************************
        Summarize for this category
        ********************************
        """

        # image_level_auroc = roc_auc_score(
        #         np.array(image_level_gt_list), np.array(image_level_pred_list)
        # )
        # logger.info(f"Image Level AUROC - {category}: {image_level_auroc}")

        pixel_level_auroc = roc_auc_score(
            np.array(pixel_level_gt_list, dtype=np.uint8),
            np.array(pixel_level_pred_list),
        )
        logger.info(f"Pixel Level AUROC - {category}: {pixel_level_auroc}")

        # image_level_auroc_all_categories.append(image_level_auroc)
        pixel_level_auroc_all_categories.append(pixel_level_auroc)

        if args.dataset == "MVTEC":
            if category in texture_types:
                # image_level_auroc_texture_categories.append(image_level_auroc)
                pixel_level_auroc_texture_categories.append(pixel_level_auroc)
            elif category in object_types:
                # image_level_auroc_object_categories.append(image_level_auroc)
                pixel_level_auroc_object_categories.append(pixel_level_auroc)

        logger.info("===========")

    """ 
    ********************************
    Summarize for all categories (get mean values)
    ********************************
    """
    # image_level_auroc_all_mean = np.mean(np.array(image_level_auroc_all_categories))
    # logger.info(f"Image Level AUROC - Mean ({len(image_level_auroc_all_categories)} classes): {image_level_auroc_all_mean}")
    pixel_level_auroc_all_mean = np.mean(np.array(pixel_level_auroc_all_categories))
    logger.info(
        f"Pixel Level AUROC - Mean ({len(pixel_level_auroc_all_categories)} classes): {pixel_level_auroc_all_mean}"
    )

    if args.dataset == "MVTEC":
        # image_level_auroc_texture_mean = np.mean(
        #     np.array(image_level_auroc_texture_categories)
        # )
        pixel_level_auroc_texture_mean = np.mean(
            np.array(pixel_level_auroc_texture_categories)
        )
        # logger.info(f"Image Level AUROC - Mean (Texture): {image_level_auroc_texture_mean}")
        logger.info(
            f"Pixel Level AUROC - Mean (Texture): {pixel_level_auroc_texture_mean}"
        )

        # image_level_auroc_object_mean = np.mean(
        #     np.array(image_level_auroc_object_categories)
        # )
        pixel_level_auroc_object_mean = np.mean(
            np.array(pixel_level_auroc_object_categories)
        )
        # logger.info(f"Image Level AUROC - Mean (Object): {image_level_auroc_object_mean}")
        logger.info(
            f"Pixel Level AUROC - Mean (Object): {pixel_level_auroc_object_mean}"
        )


if __name__ == "__main__":
    parsed_args = parse_option()

    if parsed_args.dataset == "MVTEC":
        assert parsed_args.category in mvtec_category_list + [
            "all"
        ], "Invalid category option"
        picked_classes = (
            [parsed_args.category]
            if parsed_args.category != "all"
            else mvtec_category_list
        )
    elif parsed_args.dataset == "LOCO":
        assert parsed_args.category in loco_category_list + [
            "all"
        ], "Invalid category option"
        picked_classes = (
            [parsed_args.category]
            if parsed_args.category != "all"
            else loco_category_list
        )

    logger = setup_logger(
        output=parsed_args.log_dir,
        distributed_rank=0,
        name="contrast",
        timestamp=datetime.now().strftime("%Y%m%d-%H%M%S"),
    )

    logger.info(
        f">>> device names: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}"
    )

    with torch.no_grad():
        if parsed_args.model == "CutPaste":
            nonalign_types = [
                "carpet",
                "grid",
                "leather",
                "tile",
                "wood",
                "hazelnut",
                "screw",
            ]
            eval_on_device(picked_classes, parsed_args)
        elif parsed_args.model == "DS2":
            if parsed_args.dataset == "MVTEC":
                nonalign_types = [
                    "carpet",
                    "grid",
                    "leather",
                    "tile",
                    "wood",
                ]
            elif parsed_args.dataset == "LOCO":
                nonalign_types = ["screw_bag", "splicing_connectors"]
            all_epochs = parsed_args.ckpt_epoch
            for epoch in all_epochs:
                parsed_args.ckpt_epoch = int(epoch)
                eval_on_device(picked_classes, parsed_args)
        else:
            raise ValueError(f"Invalid model option: {parsed_args.model}")
