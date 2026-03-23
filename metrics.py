# This file contains the metrics used to evaluate the performance of the model

import torch

def translation_metric(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute the L2 norm of the difference between the predicted and ground truth translations
    Based on: https://www.sciencedirect.com/science/article/pii/S0094576523003995

    Args:
        pred (torch.Tensor): Predicted translations (N x 3)
        target (torch.Tensor): Ground truth translations (N x 3)

    Returns:
        torch.Tensor: Translation metric (N)
    """
    diff = target - pred

    return torch.norm(diff, dim=1)


def relative_translation_metric(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute the relative translation metric: norm2(t_groundtruth - t_prediction) / norm2(t_groundtruth)
    Based on: https://kelvins.esa.int/satellite-pose-estimation-challenge/scoring/

    Args:
        pred (torch.Tensor): Predicted translations (N x 3)
        target (torch.Tensor): Ground truth translations (N x 3)

    Returns:
        torch.Tensor: Relative translation metric (N)
    """
    diff = target - pred
    return torch.norm(diff, dim=1) / torch.norm(target, dim=1)


def rotation_metric(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute the rotation metric: 2 * arccos(abs(<q_groundtruth, q_prediction>))
    Based on: https://kelvins.esa.int/satellite-pose-estimation-challenge/scoring/

    Args:
        pred (torch.Tensor): Predicted quaternions (N x 4)
        target (torch.Tensor): Ground truth quaternions (N x 4)

    Returns:
        torch.Tensor: Rotation metric in radians (N)
    """
    dot_product = torch.sum(pred * target, dim=1)
    epsilon = 1e-10
    dot_product = torch.clamp(torch.abs(dot_product), -1.0 + epsilon, 1.0 - epsilon)
    return 2 * torch.acos(dot_product)


def total_metric(
    pred_trans: torch.Tensor,
    target_trans: torch.Tensor,
    pred_rot: torch.Tensor,
    target_rot: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the total metric: (1/N) * sum(metric_translation + metric_rotation)

    Args:
        pred_trans (torch.Tensor): Predicted translations (N x 3)
        target_trans (torch.Tensor): Ground truth translations (N x 3)
        pred_rot (torch.Tensor): Predicted quaternions (N x 4)
        target_rot (torch.Tensor): Ground truth quaternions (N x 4)

    Returns:
        torch.Tensor: Total metric (scalar)
    """
    trans_metrics = translation_metric(pred_trans, target_trans)
    rot_metrics = rotation_metric(pred_rot, target_rot)
    return torch.mean(trans_metrics) + torch.mean(rot_metrics)


def total_relative_metric(
    pred_trans: torch.Tensor,
    target_trans: torch.Tensor,
    pred_rot: torch.Tensor,
    target_rot: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the total relative metric: (1/N) * sum(metric_translation + metric_rotation)
    Based on: https://kelvins.esa.int/satellite-pose-estimation-challenge/scoring/

    Args:
        pred_trans (torch.Tensor): Predicted translations (N x 3)
        target_trans (torch.Tensor): Ground truth translations (N x 3)
        pred_rot (torch.Tensor): Predicted quaternions (N x 4)
        target_rot (torch.Tensor): Ground truth quaternions (N x 4)

    Returns:
        torch.Tensor: Relative total metric (scalar)
    """
    relative_trans_metrics = relative_translation_metric(pred_trans, target_trans)
    rot_metrics = rotation_metric(pred_rot, target_rot)
    return torch.mean(relative_trans_metrics) + torch.mean(rot_metrics)


def compute_metrics(
    pred_trans: torch.Tensor,
    target_trans: torch.Tensor,
    pred_rot: torch.Tensor,
    target_rot: torch.Tensor,
) -> dict:
    """
    Compute all metrics and return them in a dictionary

    Args:
        pred_trans (torch.Tensor): Predicted translations (N x 3)
        target_trans (torch.Tensor): Ground truth translations (N x 3)
        pred_rot (torch.Tensor): Predicted quaternions (N x 4)
        target_rot (torch.Tensor): Ground truth quaternions (N x 4)

    Returns:
        dict: Dictionary containing all computed metrics
    """
    relative_trans_metrics = relative_translation_metric(pred_trans, target_trans)
    trans_metrics = translation_metric(pred_trans, target_trans)
    rot_metrics = rotation_metric(pred_rot, target_rot)
    total = total_metric(pred_trans, target_trans, pred_rot, target_rot)
    total_relative = total_relative_metric(pred_trans, target_trans, pred_rot, target_rot)

    return {
        "translation_metric_mean": torch.mean(trans_metrics).item(),
        "translation_metric_std": torch.std(trans_metrics).item(),
        "rotation_metric_mean": torch.mean(rot_metrics).item(),
        "rotation_metric_std": torch.std(rot_metrics).item(),
        "total_metric": total.item(),
        "relative_translation_metric_mean": torch.mean(relative_trans_metrics).item(),
        "relative_translation_metric_std": torch.std(relative_trans_metrics).item(),
        "relative_total_metric": total_relative.item(),
    }
