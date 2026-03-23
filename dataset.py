# dataset.py
# Standalone dataset class for evaluation/deployment.
# Extracted from src/datasets.py for independent use in for_deployment/.

# Author: Andrew Price
# andrew.price@epfl.ch

import json
import os
import torch
import torch.nn.functional as F
import numpy as np
from math import sqrt
from dataclasses import dataclass
from PIL import Image
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils import (
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    SPEED_Camera,
    get_apparent_orientation,
    translation_to_bbox_relative_translation
)

# ---------------------
# Keypoint Operations
# ---------------------
@dataclass
class SPEEDDataset_Keypoint_Ops():
    """
    Tracks keypoint modifications through the SPEEDDataset pipeline.

    Assumed process flow:
    1) Image crop (subtract from keypoints)
    2) Image padding to square (add to keypoints)
    3) Image resize (scale keypoints)
    """
    cx: float = 0.0
    cy: float = 0.0
    pad_x: float = 0.0
    pad_y: float = 0.0
    scale_x: float = 1.0
    scale_y: float = 1.0

    def load(self, values):
        self.cx, self.cy, self.pad_x, self.pad_y, self.scale_x, self.scale_y = values

    @property
    def values(self):
        return (self.cx, self.cy, self.pad_x, self.pad_y, self.scale_x, self.scale_y)

    def calculate_resize(self, current_size: tuple[int, int], target_size: tuple[int, int]):
        if isinstance(current_size, torch.Tensor):
            current_size = current_size.numpy() # type: ignore
        if isinstance(target_size, torch.Tensor):
            target_size = target_size.numpy() # type: ignore
        target_height, target_width = target_size
        current_height, current_width = current_size
        self.scale_x = target_width / current_width
        self.scale_y = target_height / current_height

# ---------------------
# Dataset
# ---------------------
class SPEEDDataset(Dataset):
    """
    Dataset class for the SPEED dataset (evaluation only).
    """

    def __init__(
        self,
        dataset_root_dir: str,
        split: str = "test",
        rotation_format: str = "quaternion",
        img_size: tuple[int, int] = (384, 384),
        bbox_json_path: str = None, # type: ignore
        pts3d: np.ndarray = None, # type: ignore
        args=None,
    ):
        self.dataset_root_dir = dataset_root_dir
        self.split = split
        self.rotation_format = rotation_format
        self.json_path = os.path.join(self.dataset_root_dir, f"{self.split}.json")
        self.bbox_json_path = bbox_json_path

        if args:
            self.no_pixel_augmentation = args.no_pixel_augmentation
            self.no_spatial_augmentation = args.no_spatial_augmentation
            self.no_translation_compensation = args.no_translation_compensation
            self.no_rotation_compensation = args.no_rotation_compensation
        else:
            self.no_pixel_augmentation = False
            self.no_spatial_augmentation = False
            self.no_translation_compensation = False
            self.no_rotation_compensation = False

        self.camera = SPEED_Camera()
        self.input_image_size = img_size

        # Keypoints
        if pts3d is None:
            self.pts3d = torch.tensor([
                [-0.376037, -0.381668, -0.00146],
                [-0.376037, -0.381668,  0.322591],
                [-0.376037,  0.374151, -0.00146 ],
                [-0.376037,  0.374151,  0.322591],
                [ 0.365948, -0.381668, -0.00146 ],
                [ 0.365948, -0.381668,  0.322591],
                [ 0.365948,  0.374151, -0.00146 ],
                [ 0.365948,  0.374151,  0.322591]
            ], dtype=torch.float32)
        else:
            self.pts3d = pts3d

        # Image transforms (resize + normalize + to tensor)
        self.convert_to_tensor = A.Compose(
            [
                A.Resize(height=img_size[0], width=img_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ],
            keypoint_params=A.KeypointParams(coord_format='xy', remove_invisible=False)
        )

        # Load annotations
        with open(self.json_path, "r") as f:
            self.data = json.load(f)

        if self.bbox_json_path is not None:
            with open(self.bbox_json_path, "r") as f:
                self.bbox_data = json.load(f)
        else:
            self.bbox_data = None

    def __len__(self) -> int:
        return len(self.data)

    def crop_image(self, image, bbox, pts2d, kpt_ops):
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cropped_image = image[y1:y2, x1:x2, :]

        if pts2d is not None:
            pts2d = pts2d - np.array([[x1, y1]])
            kpt_ops.cx = x1
            kpt_ops.cy = y1

        h, w, _ = cropped_image.shape
        if h > w:
            pad = (h - w) // 2
            cropped_image = np.pad(cropped_image, ((0, 0), (pad, pad), (0, 0)), mode="constant")
            if pts2d is not None:
                pts2d[:, 0] += pad
                kpt_ops.pad_x = pad
        elif w > h:
            pad = (w - h) // 2
            cropped_image = np.pad(cropped_image, ((pad, pad), (0, 0), (0, 0)), mode="constant")
            if pts2d is not None:
                pts2d[:, 1] += pad
                kpt_ops.pad_y = pad

        return cropped_image, pts2d, kpt_ops

    def project_points(self, pts3d, K, R, T):
        T = T.reshape(3, 1)
        pts3d = np.matmul(R, pts3d.T)
        pts3d = pts3d + T
        pts2d = np.matmul(K, pts3d)
        pts2d = pts2d[:2] / pts2d[2]
        return pts2d.T

    def __getitem__(self, idx: int):
        item = self.data[idx]
        filename = item["filename"]
        img_name = os.path.join(self.dataset_root_dir, "images", self.split, filename)

        image = np.array(Image.open(img_name).convert("RGB"))

        # Bbox
        if self.bbox_json_path:
            bbox_info = self.bbox_data.get(filename) # type: ignore
            if bbox_info is None:
                raise ValueError(f"No bbox information found for image {filename}")
            bbox = [bbox_info["x1"], bbox_info["y1"], bbox_info["x2"], bbox_info["y2"]]
        else:
            bbox = None

        # Pose
        quaternion = torch.tensor(item["q_vbs2tango"], dtype=torch.float32)
        translation = torch.tensor(item["r_Vo2To_vbs_true"], dtype=torch.float32)

        quaternion = get_apparent_orientation(
            translation=translation,
            centered_rotation_quat=quaternion,
            no_rotation_compensation=self.no_rotation_compensation,
        )

        rotation = quaternion

        # 2D keypoints
        kpt_ops = SPEEDDataset_Keypoint_Ops()
        pts2d = self.project_points(
            self.pts3d,
            self.camera.K,
            quaternion_to_rotation_matrix(rotation),
            translation
        )

        # Crop
        if self.bbox_json_path:
            image, pts2d, kpt_ops = self.crop_image(image, bbox, pts2d, kpt_ops)

        # Resize + normalize
        kpt_ops.calculate_resize(
            current_size=(image.shape[0], image.shape[1]),
            target_size=self.input_image_size
        )
        transformed = self.convert_to_tensor(image=image, keypoints=pts2d)
        pts2d = transformed['keypoints']
        image = transformed["image"]

        if self.bbox_json_path:
            bbox = torch.tensor(bbox, dtype=torch.float32)
            translation = translation_to_bbox_relative_translation(
                translation, bbox, self.camera, self.no_translation_compensation
            )
        else:
            bbox = torch.zeros(4, dtype=torch.float32)

        if self.rotation_format == "matrix":
            rotation = quaternion_to_rotation_matrix(rotation)

        return {
            'image': image,
            'K': self.camera.K,
            'translation': translation,
            'rotation': rotation,
            'bbox': bbox,
            'pts2d': pts2d,
            'pts3d': self.pts3d,
            'kpt_ops': torch.tensor(kpt_ops.values, dtype=image.dtype)
        }
