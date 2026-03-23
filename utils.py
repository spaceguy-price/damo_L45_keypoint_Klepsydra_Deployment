# utils.py
# This script has been simplified from a larger evaluation repository.
# It is contains necessary functions from src.engine, src.keypoints, src.utils
# It is intended to support the Klepsydra deployment process

# Author: Andrew Price
# andrew.price@epfl.ch
# 19.03.2026

from dataclasses import dataclass, field
from math import sqrt
import numpy as np
import cv2
import torch
import torch.nn.functional as F

# ------------------------------
# PnP Utilities
# ------------------------------
def perform_PnP(
    pts2d: torch.tensor,        # type: ignore
    pts3d: torch.tensor,        # type: ignore
    intrinsics: torch.tensor,   # type: ignore
) -> dict[str, torch.Tensor]:
    """
    Perform PnP on estimated 2D keypoints. This operation is performed on the CPU.

    Args:
        pts2d (torch.tensor): [B,N,2] Predicted 2D keypoints (batch size B, number of keypoints N)
        pts3d (torch.tensor): [N,3] Model keypoints
        K (torch.tensor): [B,3,3] Camera intrinsic matrix

    Returns:
        Dictionary containing the predicted translations, rotations (rotation matrices), and success flags.
    """

    pred_translations = []
    pred_rotations = []
    succeeded = []

    if isinstance(pts2d, torch.Tensor):
        pts2d = pts2d.numpy()
    if isinstance(pts3d, torch.Tensor):
        pts3d = pts3d.numpy()
    if isinstance(intrinsics, torch.Tensor):
        intrinsics = intrinsics.numpy()

    for i in range(pts2d.shape[0]): #Loop through the batch
        # Extra predictions and dataloader ground truths
        kp2d = pts2d[i]
        kp3d = pts3d[i]
        K = intrinsics[i]
        
        # Run PnP
        success, rvec, tvec = cv2.solvePnP(
            objectPoints=kp3d,
            imagePoints=kp2d,
            cameraMatrix=K,
            distCoeffs=None, #type: ignore
            flags=cv2.SOLVEPNP_EPNP
        ) 

        if not success:
            rvec = np.array([0,0,0]).astype(np.float32)
            tvec = np.array([0,0,0]).astype(np.float32)

        # Rodrigues rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)

        # Collect results
        pred_translations.append(torch.tensor(tvec.squeeze()))
        pred_rotations.append(torch.tensor(R))              
        succeeded.append(torch.tensor(success))
        
    pred_translations = torch.stack(pred_translations, dim=0) #[B,3]
    pred_rotations = torch.stack(pred_rotations, dim=0) #[B,3,3]
    succeeded = torch.stack(succeeded, dim=0) #[B,1]

    preds_PnP = {'pred_translations': pred_translations,
                'pred_rotations': pred_rotations,
                'successes': succeeded
                }

    return preds_PnP

def restore_keypoints_batch(
        kpts_ops, kpts_batch
        ) -> torch.Tensor:
    """
    Restore keypoints to the uncropped, unpadded, unresized image

    Args:
        kps_ops (torch.Tensor): (B,6) -> (B,[cx, cy, pad_x, pad_y, scale_x, scale_y])
            Keypoint opertions
        kpts_batch (torch.Tensor): (B,N,2) Cropped image keypoints

    Returns:
        kpts_batch (torch.Tensor): (B,N,2) Image keypoints for original unmodified image

    B - batch size
    N - number of keypoints
    """
    #Undo image resizing
    scale = kpts_ops[:,4:] # Obtain scaling parameters (B, 2)
    kpts_batch /= torch.unsqueeze(scale, 1) # Unscale keypoints by the scale factor

    #Undo padding
    pad = kpts_ops[:,2:4] # Obtain the padding parameters (B, 2)
    kpts_batch -= torch.unsqueeze(pad, 1)  # Remove the effect of padding

    #Undo cropping
    offset = kpts_ops[:,:2] # Obtain the bounding box offset parameters (B, 2)
    kpts_batch += torch.unsqueeze(offset, 1)  # Remove the effect of cropping

    return kpts_batch

SPEED_pts3d = torch.tensor([
    [-0.376037, -0.381668, -0.00146],
    [-0.376037, -0.381668,  0.322591],
    [-0.376037,  0.374151, -0.00146 ],
    [-0.376037,  0.374151,  0.322591],
    [ 0.365948, -0.381668, -0.00146 ],
    [ 0.365948, -0.381668,  0.322591],
    [ 0.365948,  0.374151, -0.00146 ],
    [ 0.365948,  0.374151,  0.322591]
    ], dtype=torch.float32)

# ------------------------------
# Math and Pose Utilities
# ------------------------------
def rotation_matrix_to_quaternion(
        rotation_matrix: torch.Tensor
        ) -> torch.Tensor:
    """
    Convert a batch of 3x3 rotation matrices to quaternions. Treats multiple cases for numerical stability.
    Based on: https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/

    Args:
        rotation_matrix (torch.Tensor): Batch of rotation matrices (N x 3 x 3)

    Returns:
        torch.Tensor: Batch of quaternions (N x 4) in format (w, x, y, z)
    """
    rotation_matrix = rotation_matrix.float() # Ensure rotation matrix is a float for upcoming operations
    # Check if input is batched
    unbatched = rotation_matrix.dim() == 2
    if unbatched:
        rotation_matrix = rotation_matrix.unsqueeze(0)

    batch_size = rotation_matrix.size(0)

    # Compute trace of rotation matrix
    trace = rotation_matrix[:, 0, 0] + rotation_matrix[:, 1, 1] + rotation_matrix[:, 2, 2]

    # Initialize quaternions
    q = torch.zeros(batch_size, 4, device=rotation_matrix.device)

    # Case 1: Trace > 0
    mask_1 = trace > 0
    r = torch.sqrt(torch.clamp(1 + trace[mask_1], min=1e-8))
    s = 0.5 / r
    q[mask_1, 0] = 0.5 * r
    q[mask_1, 1] = (rotation_matrix[mask_1, 2, 1] - rotation_matrix[mask_1, 1, 2]) * s
    q[mask_1, 2] = (rotation_matrix[mask_1, 0, 2] - rotation_matrix[mask_1, 2, 0]) * s
    q[mask_1, 3] = (rotation_matrix[mask_1, 1, 0] - rotation_matrix[mask_1, 0, 1]) * s

    # Case 2: R[0,0] > R[1,1] and R[0,0] > R[2,2]
    mask_2 = (
        (~mask_1)
        & (rotation_matrix[:, 0, 0] > rotation_matrix[:, 1, 1])
        & (rotation_matrix[:, 0, 0] > rotation_matrix[:, 2, 2])
    )
    r = torch.sqrt(
        torch.clamp(
            1
            + rotation_matrix[mask_2, 0, 0]
            - rotation_matrix[mask_2, 1, 1]
            - rotation_matrix[mask_2, 2, 2],
            min=1e-8,
        )
    )
    s = 0.5 / r
    q[mask_2, 1] = 0.5 * r
    q[mask_2, 0] = (rotation_matrix[mask_2, 2, 1] - rotation_matrix[mask_2, 1, 2]) * s
    q[mask_2, 2] = (rotation_matrix[mask_2, 0, 1] + rotation_matrix[mask_2, 1, 0]) * s
    q[mask_2, 3] = (rotation_matrix[mask_2, 0, 2] + rotation_matrix[mask_2, 2, 0]) * s

    # Case 3: R[1,1] > R[2,2]
    mask_3 = (~mask_1) & (~mask_2) & (rotation_matrix[:, 1, 1] > rotation_matrix[:, 2, 2])
    r = torch.sqrt(
        torch.clamp(
            1
            - rotation_matrix[mask_3, 0, 0]
            + rotation_matrix[mask_3, 1, 1]
            - rotation_matrix[mask_3, 2, 2],
            min=1e-8,
        )
    )
    s = 0.5 / r
    q[mask_3, 2] = 0.5 * r
    q[mask_3, 0] = (rotation_matrix[mask_3, 0, 2] - rotation_matrix[mask_3, 2, 0]) * s
    q[mask_3, 1] = (rotation_matrix[mask_3, 0, 1] + rotation_matrix[mask_3, 1, 0]) * s
    q[mask_3, 3] = (rotation_matrix[mask_3, 1, 2] + rotation_matrix[mask_3, 2, 1]) * s

    # Case 4: R[2,2] is largest diagonal term
    mask_4 = (~mask_1) & (~mask_2) & (~mask_3)
    r = torch.sqrt(
        torch.clamp(
            1
            - rotation_matrix[mask_4, 0, 0]
            - rotation_matrix[mask_4, 1, 1]
            + rotation_matrix[mask_4, 2, 2],
            min=1e-8,
        )
    )
    s = 0.5 / r
    q[mask_4, 3] = 0.5 * r
    q[mask_4, 0] = (rotation_matrix[mask_4, 1, 0] - rotation_matrix[mask_4, 0, 1]) * s
    q[mask_4, 1] = (rotation_matrix[mask_4, 0, 2] + rotation_matrix[mask_4, 2, 0]) * s
    q[mask_4, 2] = (rotation_matrix[mask_4, 1, 2] + rotation_matrix[mask_4, 2, 1]) * s

    # Normalize quaternions
    q = q / torch.norm(q, dim=1, keepdim=True)

    # Return squeezed output if input was unbatched
    return q.squeeze(0) if unbatched else q

def quaternion_to_rotation_matrix(
        q: torch.Tensor
        ) -> torch.Tensor:
    """
    Converts a unit quaternion to a 3x3 rotation matrix. Based on:
    https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation

    Args:
        q (torch.Tensor): Unit quaternion (4,)

    Returns:
        torch.Tensor: Rotation matrix (3, 3)
    """
    qr, qi, qj, qk = q[0], q[1], q[2], q[3]

    # Compute the rotation matrix elements
    r00 = 1 - 2 * (qj**2 + qk**2)
    r01 = 2 * (qi * qj - qk * qr)
    r02 = 2 * (qi * qk + qj * qr)

    r10 = 2 * (qi * qj + qk * qr)
    r11 = 1 - 2 * (qi**2 + qk**2)
    r12 = 2 * (qj * qk - qi * qr)

    r20 = 2 * (qi * qk - qj * qr)
    r21 = 2 * (qj * qk + qi * qr)
    r22 = 1 - 2 * (qi**2 + qj**2)

    # Form the rotation matrix
    rotation_matrix = torch.stack(
        [
            torch.tensor([r00, r01, r02]),
            torch.tensor([r10, r11, r12]),
            torch.tensor([r20, r21, r22]),
        ]
    )

    return rotation_matrix

def batched_quaternion_to_rotation_matrix(
        q: torch.Tensor
        ) -> torch.Tensor:
    """
    Converts batched unit quaternions to 3x3 rotation matrices. Based on:
    https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation

    Args:
        q (torch.Tensor): Batch of unit quaternions (B, 4)

    Returns:
        torch.Tensor: Batch of rotation matrices (B, 3, 3)
    """
    # Extract quaternion components
    qr = q[..., 0]
    qi = q[..., 1]
    qj = q[..., 2]
    qk = q[..., 3]

    # Compute the rotation matrix elements
    r00 = 1 - 2 * (qj**2 + qk**2)
    r01 = 2 * (qi * qj - qk * qr)
    r02 = 2 * (qi * qk + qj * qr)

    r10 = 2 * (qi * qj + qk * qr)
    r11 = 1 - 2 * (qi**2 + qk**2)
    r12 = 2 * (qj * qk - qi * qr)

    r20 = 2 * (qi * qk - qj * qr)
    r21 = 2 * (qj * qk + qi * qr)
    r22 = 1 - 2 * (qi**2 + qj**2)

    # Stack the elements into a batch of 3x3 matrices
    rotation_matrix = torch.stack(
        [
            torch.stack([r00, r01, r02], dim=-1),
            torch.stack([r10, r11, r12], dim=-1),
            torch.stack([r20, r21, r22], dim=-1),
        ],
        dim=-2,
    )

    return rotation_matrix

def get_apparent_orientation(
    translation: torch.Tensor,
    centered_rotation_quat: torch.Tensor,
    no_rotation_compensation: bool = False,
) -> torch.Tensor:
    """
    Convert centered rotation to apparent rotation (perturbed by translation T).
    """
    if no_rotation_compensation:
        return centered_rotation_quat

    R_centered = quaternion_to_rotation_matrix(centered_rotation_quat)
    v_new = F.normalize(translation, p=2, dim=0)
    v_old = torch.tensor([0.0, 0.0, 1.0], device=translation.device)

    axis = torch.cross(v_old, v_new, dim=0)
    if torch.all(axis == 0):
        if torch.dot(v_old, v_new) > 0:
            return centered_rotation_quat
        else:
            return torch.tensor([1.0, 0.0, 0.0, 0.0], device=translation.device)

    axis = F.normalize(axis, p=2, dim=0)
    angle = torch.acos(torch.dot(v_old, v_new))

    K = torch.tensor(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]],
        device=translation.device,
    )
    R_rel = (
        torch.eye(3, device=translation.device)
        + torch.sin(angle) * K
        + (1 - torch.cos(angle)) * (K @ K)
    )

    R_original = R_rel.t() @ R_centered
    return rotation_matrix_to_quaternion(R_original)

def batched_get_absolute_orientation(
    translation: torch.Tensor, rotation_quat: torch.Tensor, no_rotation_compensation: bool = False
) -> torch.Tensor:
    """
    Convert the absolute rotation (orientation when the object is centered on the image) to the apparent rotation (orientation perturbed by translation T) - batched version.

    Args:
        translation (torch.Tensor): (B, 3) tensor of translation vectors.
        rotation_quat (torch.Tensor): (B, 4) tensor of quaternions representing rotations.
        no_rotation_compensation: bool to disable rotation compensation

    Returns:
        torch.Tensor: (B, 4) tensor of centered rotations as quaternions.
    """
    # Return the original rotation if no rotation compensation is desired
    if no_rotation_compensation:
        return rotation_quat

    batch_size = translation.shape[0]
    device = translation.device

    # Convert quaternions to rotation matrices (B, 3, 3)
    R_initial = batched_quaternion_to_rotation_matrix(rotation_quat).float()

    # Normalize the translation vectors (B, 3)
    v_new = F.normalize(translation, p=2, dim=1)

    # Original viewing direction (expanded to batch size)
    v_old = torch.tensor([0.0, 0.0, 1.0], device=device).expand(batch_size, 3)

    # Compute rotation axis and angle (B, 3)
    axis = torch.cross(v_old, v_new, dim=1)

    # Compute angles (B,)
    angles = torch.acos(torch.clamp(torch.sum(v_old * v_new, dim=1), -1.0, 1.0))  # Clamp

    # Normalize the axis to avoid division by zero
    axis_norm = torch.norm(axis, dim=1, keepdim=True)
    axis = axis / torch.clamp(axis_norm, min=1e-6)

    # Handle cases where vectors are parallel or anti-parallel
    parallel_mask = (axis_norm < 1e-6).squeeze(-1)  # True if parallel or anti-parallel
    parallel_sign = torch.sign(
        torch.sum(v_old * v_new, dim=1)
    )  # +1 for parallel, -1 for anti-parallel

    # Compute Rodrigues' rotation matrices
    K = torch.zeros(batch_size, 3, 3, device=device)
    K[:, 0, 1] = -axis[:, 2]
    K[:, 0, 2] = axis[:, 1]
    K[:, 1, 0] = axis[:, 2]
    K[:, 1, 2] = -axis[:, 0]
    K[:, 2, 0] = -axis[:, 1]
    K[:, 2, 1] = axis[:, 0]

    identity = torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1)
    sin_angles = torch.sin(angles).unsqueeze(-1).unsqueeze(-1)
    cos_angles = torch.cos(angles).unsqueeze(-1).unsqueeze(-1)

    R_rel = identity + sin_angles * K + (1 - cos_angles) * (K @ K)

    # Handle parallel and anti-parallel cases
    R_parallel = torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1)
    R_antiparallel = (
        torch.diag(torch.tensor([1.0, -1.0, -1.0], device=device))
        .unsqueeze(0)
        .expand(batch_size, -1, -1)
    )

    R_rel = torch.where(
        parallel_mask.unsqueeze(-1).unsqueeze(-1),
        torch.where((parallel_sign > 0).unsqueeze(-1).unsqueeze(-1), R_parallel, R_antiparallel),
        R_rel,
    )

    # Apply the relative rotation to the initial rotation
    R_new = R_rel @ R_initial

    # Convert the resulting rotation matrices back to quaternions
    return rotation_matrix_to_quaternion(R_new)

# ------------------------------
# Dataset Utilities
# ------------------------------
@dataclass
class SPEED_Camera():
    """
    Data class modified from UrsoNet's implementation:
    https://github.com/pedropro/UrsoNet/blob/8e59d9b81dd3805aba1d773bd9b44f1a33745b05/speed.py
    Parameters verified with the original SPEED dataset paper:
    https://arxiv.org/pdf/1906.09868
    """
    fwx: float = 0.0176     # Focal length[m]
    fwy: float = 0.0176
    width: int = 1920       # Image size [pixels]
    height: int = 1200
    ppx: float = 5.86e-6    # Size of the pixels [m / pixel]
    ppy: float = 5.86e-6
    _K: torch.Tensor = field(init=False, repr=False) #Don't include this in the __init__ (it should be calculated based on user inputs)

    def __post_init__(self):
        self._K = torch.tensor(
            [[self.fx, 0, self.width / 2],
             [0, self.fy, self.height / 2],
             [0,0,1]]
        )
        self._K_original = self._K

    # Focal length[pixels]
    @property
    def fx(self):
        return self.fwx / self.ppx
    @property
    def fy(self):
        return self.fwy / self.ppy

    # Intrinsics matrix
    @property
    def K(self):
        return self._K.clone()
    @property
    def K_inv(self):
        return torch.inverse(self._K.clone())
    @property
    def aspect_ratio(self):
        return self.width / self.height

# ------------------------------
# Bounding Box Utilities
# ------------------------------
def batched_bbox_relative_translation_to_translation(
    bbox_relative_translation: torch.Tensor,
    bbox: torch.Tensor,
    camera: SPEED_Camera,
    no_translation_compensation: bool = False,
) -> torch.Tensor:
    """
    Transform the bbox relative translation vectors to absolute translations based on the bounding boxes and camera parameters.
    Handles batched inputs.
    Inspired by: https://openaccess.thecvf.com/content_ICCV_2019/papers/Li_CDPN_Coordinates-Based_Disentangled_Pose_Network_for_Real-Time_RGB-Based_6-DoF_Object_ICCV_2019_paper.pdf

    Args:
        bbox_relative_translation: torch.Tensor of shape (B, 3) where B is the batch size
        bbox: torch.Tensor of shape (B, 4) containing [x1, y1, x2, y2] for each box
        camera: Camera object with intrinsic parameters
        no_translation_compensation: bool to disable translation compensation

    Returns:
        torch.Tensor of shape (B, 3) containing absolute translations
    """
    # Compute the width and height of the bounding boxes
    w = bbox[:, 2] - bbox[:, 0]  # Shape: (B,)
    h = bbox[:, 3] - bbox[:, 1]  # Shape: (B,)

    # Compute the centers of the bounding boxes
    bbox_center_x = bbox[:, 0] + w / 2  # Shape: (B,)
    bbox_center_y = bbox[:, 1] + h / 2  # Shape: (B,)

    # Image center (broadcast to batch size)
    cx = camera.width / 2
    cy = camera.height / 2

    # Compute the maximum ratio between the width and height of the bounding boxes
    # Divide by aspect ratio to better the resizing process
    avg_ratio = torch.sqrt(
        camera.width * camera.height / (w * h * camera.aspect_ratio)
    )  # Shape: (B,)
    # avg_ratio = torch.sqrt(camera.width * camera.height / (w * h))  # Shape: (B,)
    # avg_ratio = torch.min(camera.width/w, camera.height/h)  # Shape: (B,)

    A = (bbox_relative_translation[:, 0] * w - cx + bbox_center_x) / camera.fx  # Shape: (B,)
    B = (bbox_relative_translation[:, 1] * h - cy + bbox_center_y) / camera.fy  # Shape: (B,)

    if no_translation_compensation:
        translation_z = avg_ratio * bbox_relative_translation[:, 2]
    else:
        translation_z = (
            avg_ratio * torch.sqrt(A**2 + B**2 + 1) * bbox_relative_translation[:, 2]
        )  # Shape: (B,)
    translation_x = A * translation_z  # Shape: (B,)
    translation_y = B * translation_z  # Shape: (B,)

    # # We want to recover Tz from Tz/max_ratio
    # translation_z = bbox_relative_translation[:, 2] * avg_ratio  # Shape: (B,)

    # # Recover absolute translations
    # translation_x = (
    #     (bbox_relative_translation[:, 0] * w - cx + bbox_center_x) * translation_z / camera.fx
    # )  # Shape: (B,)
    # translation_y = (
    #     (bbox_relative_translation[:, 1] * h - cy + bbox_center_y) * translation_z / camera.fy
    # )  # Shape: (B,)

    # Stack the results into a (B, 3) tensor
    translation = torch.stack([translation_x, translation_y, translation_z], dim=1)

    return translation

def translation_to_bbox_relative_translation(
    translation: torch.Tensor,
    bbox: torch.Tensor,
    camera: SPEED_Camera,
    no_translation_compensation: bool = False,
) -> torch.Tensor:
    """
    Transform translation to bbox-relative coordinates.
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    bbox_center_x = bbox[0] + w / 2
    bbox_center_y = bbox[1] + h / 2

    cx = camera.width / 2
    cy = camera.height / 2

    avg_ratio = sqrt(camera.width * camera.height / (w * h * camera.aspect_ratio))

    new_target_z = translation[2] / avg_ratio
    if not no_translation_compensation:
        new_target_z = new_target_z * sqrt(
            (translation[0] ** 2 + translation[1] ** 2 + translation[2] ** 2)
            / (translation[2] ** 2)
        )

    new_target_x = ((translation[0] * camera.fx / translation[2]) + cx - bbox_center_x) / w
    new_target_y = ((translation[1] * camera.fy / translation[2]) + cy - bbox_center_y) / h
    return torch.tensor([new_target_x, new_target_y, new_target_z], dtype=torch.float32)

