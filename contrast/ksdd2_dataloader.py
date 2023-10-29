import os
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
import glob
from contrast.ksdd2_train_normal import ksdd2_train_normal_image_paths


class KSDD2TestDataset(Dataset):
    def __init__(self, root_dir, resize_shape=None):
        self.root_dir = root_dir
        self.gt_images = sorted(glob.glob(root_dir + "*_GT.png"))
        self.raw_images = ["".join(item.split("_GT")) for item in self.gt_images]
        self.images = list(zip(self.raw_images, self.gt_images))
        self.resize_shape = resize_shape

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.resize_shape != None:
            image = cv2.resize(
                image, dsize=(self.resize_shape[1], self.resize_shape[0])
            )
            mask = cv2.resize(mask, dsize=(self.resize_shape[1], self.resize_shape[0]))

        image = image / 255.0
        mask = mask / 255.0

        image = (
            np.array(image)
            .reshape((image.shape[0], image.shape[1], 3))
            .astype(np.float32)
        )
        mask = (
            np.array(mask).reshape((mask.shape[0], mask.shape[1], 1)).astype(np.float32)
        )

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        return image, mask

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        (raw_img_path, gt_img_path) = self.images[idx]
        dir_path, file_name = os.path.split(raw_img_path)

        image, mask = self.transform_image(raw_img_path, gt_img_path)
        has_anomaly = np.array([1 if 1 in mask else 0], dtype=np.float32)

        sample = {
            "image": image,
            "has_anomaly": has_anomaly,
            "mask": mask,
            "idx": idx,
            "file_name": file_name.split(".")[0],
        }

        return sample


class KSDD2TrainDataset(Dataset):
    def __init__(self, resize_shape=None):
        self.resize_shape = resize_shape
        self.image_paths = ksdd2_train_normal_image_paths

    def __len__(self):
        return len(self.image_paths)

    def transform_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
        image = (
            np.array(image)
            .reshape((image.shape[0], image.shape[1], image.shape[2]))
            .astype(np.float32)
            / 255.0
        )
        image = np.transpose(image, (2, 0, 1))
        return image

    def __getitem__(self, idx):
        image = self.transform_image(self.image_paths[idx])

        sample = {
            "image": image,
            "idx": idx,
        }

        return sample
