import os
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
import glob


class MVTecTestDataset(Dataset):
    def __init__(self, root_dir, resize_shape=None):
        self.root_dir = root_dir
        self.images = sorted(glob.glob(root_dir + "/*/*.png"))
        self.resize_shape = resize_shape

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]))
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

        # before transpose, e.g., image.shape: (256, 256, 3)
        # after transpose, e.g., image.shape: (3, 256, 256)
        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        return image, mask

    def __getitem__(self, idx):
        # e.g., idx: 0
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # e.g., img_path: ./datasets/mvtec/capsule/test/crack/000.png
        img_path = self.images[idx]
        # e.g., dir_path: ./datasets/mvtec/capsule/test/crack
        # e.g., file_name: 000.png
        dir_path, file_name = os.path.split(img_path)
        # e.g., base_dir: crack
        base_dir = os.path.basename(dir_path)

        if base_dir == "good":
            image, mask = self.transform_image(img_path, None)
            has_anomaly = np.array(0, dtype=np.float32)
        else:
            # e.g., mask_path: ./datasets/mvtec/capsule/ground_truth/
            mask_path = os.path.join(dir_path, "../../ground_truth/")
            # e.g., mask_path: ./datasets/mvtec/capsule/ground_truth/crack
            mask_path = os.path.join(mask_path, base_dir)
            # e.g., mask_file_name: 000_mask.png
            mask_file_name = file_name.split(".")[0] + "_mask.png"
            # e.g., mask_path: ./datasets/mvtec/capsule/ground_truth/crack/000_mask.png
            mask_path = os.path.join(mask_path, mask_file_name)
            image, mask = self.transform_image(img_path, mask_path)
            has_anomaly = np.array(1, dtype=np.float32)

        sample = {
            "image": image,
            "has_anomaly": has_anomaly,
            "mask": mask,
            "idx": idx,
            "file_name": base_dir + "_" + file_name.split(".")[0],
        }

        return sample


class MVTecTrainDataset(Dataset):
    def __init__(self, root_dir, resize_shape=None):
        """
        Args:
            root_dir (string): Directory with all the images.
        """
        self.root_dir = root_dir
        self.resize_shape = resize_shape
        self.image_paths = sorted(glob.glob(root_dir + "/*.png"))

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
        # idx = torch.randint(0, len(self.image_paths), (1,)).item()

        image = self.transform_image(self.image_paths[idx])

        sample = {
            "image": image,
            "idx": idx,
        }

        return sample
