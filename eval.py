# eval.py
# This script has been simplified from a larger evaluation repository.
# It is intended to support the Klepsydra deployment process but does NOT contain:
#  different-models: [backbones, heads, activations]; or 
#  optimizations: [quantization, sparsity, pruning]; or
#  experiments: [SPEED dataloader, fault injection, data augmentation, logging]

# Author: Andrew Price
# andrew.price@epfl.ch
# 17.03.2026

import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse

from utils import (
    perform_PnP,
    restore_keypoints_batch,
    rotation_matrix_to_quaternion, 
    batched_bbox_relative_translation_to_translation,
    batched_get_absolute_orientation
    ) 
from metrics import compute_metrics
from model import DamoPose

# These imports are only needed for evaluation mode (args.mode == "evaluation")
import json
from dataset import SPEEDDataset
from tqdm import tqdm
# These imports are only needed for single image inference mode (args.mode == "single")
import json
from PIL import Image
from utils import SPEED_pts3d, SPEED_Camera

def process_commandline_inputs(
        return_parser=False
        ):
    """
    Builds the parser and returns the arguments (or the parser itself)

    Args:
        return_parser (bool): If True, returns the argparse.ArgumentParser object instead of parsed args.

    Returns:
        args (argparse.Namespace) or parser (argparse.ArgumentParser)
    """
    parser = argparse.ArgumentParser(
        description="Evaluate SimplePose on SPEED dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    #-----------------------
    # Training or Inference?
    #-----------------------
    parser.add_argument(
        "--mode",
        type=str,
        choices = ["training", "evaluation", "single"],
        help = "training, dataset evaluation, or single image inference?"
    )

    #-------------------------
    # Path and Data Parameters
    #-------------------------
    parser.add_argument(
        "--dataset_root_dir",
        type=str,
        default="./SPEED_FIXED",
        help="Root directory of the SPEED dataset",
    )
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=4, 
        help="Number of workers for data loading"
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default=None,
        help="Path to save the best model",
    )
    parser.add_argument(
        "--bbox_json_path",
        type=str,
        default=None,
        help="Path to the JSON file containing bounding box information. If this path is provided, the model will use bounding boxes, otherwise it will use the full image",
    )
    parser.add_argument(
        "--single_image_path",
        type=str,
        default=None,
        help="Path to a single image for inference when --mode=single",
    )
    parser.add_argument(
        "--single_gt_path",
        type=str,
        default=None,
        help="Path to a single JSON file containing the ground truth translation and rotation for the image specified by --single_image_path when --mode=single",
    )

    #-----------------
    # Model Parameters
    #-----------------
    parser.add_argument(
        "--model_backbone",
        type=str,
        default="damo_L45_L",
        help="Model backbone to use",
    )
    parser.add_argument(
        "--head_type",
        type=str,
        choices=["direct_regression", "keypoint"],
        default="keypoint",
        help="The type of model head to construct",
    )
    parser.add_argument(
        "--model_weights", 
        type=str, 
        default=None, 
        help="Path to pretrained full model weights"
    )
    parser.add_argument(
        "--num_hidden_layers",
        type=int,
        default=3,
        help="Number of hidden layers in MLPWithProjection",
    )
    parser.add_argument(
        "--hidden_layer_dim",
        type=int,
        default=800,
        help="Dimension of hidden layers in MLPWithProjection",
    )

    #-----------------------------
    # CNN (DAMO-specific) checkpoint/load options
    #-----------------------------
    parser.add_argument(
        "--damo_load_verbose",
        action="store_true",
        help="Print detailed info about which checkpoint keys are loaded/skipped.",
    )
    parser.add_argument(
        "--damo_load_strict",
        action="store_true",
        help="Fail if skipped keys are not in allowed prefixes or if load ratio below threshold.",
    )
    parser.add_argument(
        "--damo_min_load_ratio",
        type=float,
        default=0.6,
        help="Minimum ratio of loaded keys required; used with strict mode or warned otherwise.",
    )
    parser.add_argument(
        "--damo_allowed_skip_prefixes",
        type=str,
        default="head.",
        help="Comma-separated prefixes allowed to be skipped when loading checkpoints (e.g., head.).",
    )
    parser.add_argument(
        "--damo_shape_verbose",
        action="store_true",
        help="Print inferred FPN channels and spatial sizes for CNN backbones.",
    )

    #---------------------
    # Inference Parameters
    #---------------------   
    parser.add_argument(
        "--output_json",
        type=str,
        help="Path to save the predictions and ground-truth in JSON format (when using evaluate.py)",
    )

    #--------------------
    # Training parameters
    #--------------------
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=20, 
        help="Number of epochs to train"
        )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=8, 
        help="Batch size for training and validation"
    )
    parser.add_argument(
        "--max_lr", 
        type=float, 
        default=1e-5, 
        help="Maximum learning rate for cosine annealing"
    )
    parser.add_argument(
        "--min_lr", 
        type=float, 
        default=1e-6, 
        help="Minimum learning rate for cosine annealing"
    )
    parser.add_argument(
        "--translation_loss",
        type=str,
        choices=["mse", "relative_mse"],
        default="mse",
        help="Loss function for translation",
    )
    parser.add_argument(
        "--rotation_format",
        type=str,
        choices=["quaternion", "matrix"],
        default="quaternion",
        help="Format for representing rotations",
    )
    parser.add_argument(
        "--normalize_quaternions",
        action="store_true",
        help="Normalize predicted quaternions during evaluation when rotation_format=quaternion",
    )
    parser.add_argument(
        "--rotation_loss",
        type=str,
        choices=["mse", "geodesic", "simplified", "frobenius"],
        default="mse",
        help="Loss function for rotation",
    )
    parser.add_argument(
        "--keypoint_loss",
        type=str,
        choices=["smoothL1", "mse", "L1", "3DsmoothL1"],
        default="smoothL1",
        help="Loss function for keypoints"
    )
    parser.add_argument(
        "--penalize_bad_basis",
        action="store_true",
        help="Penalize bad basis in the rotation matrix or quaternion",
    )
    parser.add_argument(
        "--no_translation_compensation",
        action="store_true",
        help="Disable translation compensation",
    )
    parser.add_argument(
        "--no_rotation_compensation",
        action="store_true",
        help="Disable rotation compensation",
    )

    if return_parser:
        return parser
    return parser.parse_args()

def obtain_damoL45_kpt(
        args: argparse.Namespace,
) -> tuple[torch.nn.Module, tuple[int, int]]:
    """
    Obtain the DAMO L45 L backbone with keypoint head model.
    This specific function was written to support the Klepsydra deployment process.

    Args:
        args (argparse.Namespace): Command-line arguments containing model parameters

    Returns:
        model (torch.nn.Module): The constructed model
        image_size (int, int): The expected input image size for the model (height, width)
    """

    image_size = (384, 384) # The DAMO L45 L backbone with keypoint head was trained on 384x384 images

    model = DamoPose(
            num_hidden_layers=args.num_hidden_layers,
            hidden_layer_dim=args.hidden_layer_dim,
            damo_load_verbose=args.damo_load_verbose,
            damo_load_strict=args.damo_load_strict,
            damo_min_load_ratio=args.damo_min_load_ratio,
            damo_allowed_skip_prefixes=args.damo_allowed_skip_prefixes,
            damo_shape_verbose=args.damo_shape_verbose,
        )
    # Load model weights
    if args.model_weights is not None:
        checkpoint = torch.load(args.model_weights, map_location='cpu', weights_only=True)
        if "model_state_dict" in checkpoint:
            try:
                model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            except RuntimeError as e:
                # Fallback: filter out incompatible shapes (e.g., different head dims)
                current_sd = model.state_dict()
                filtered = {
                    k: v
                    for k, v in checkpoint["model_state_dict"].items()
                    if k in current_sd and isinstance(v, torch.Tensor) and v.shape == current_sd[k].shape
                }
                skipped = len(checkpoint["model_state_dict"]) - len(filtered)
                print(f"[weights] load_state_dict shape mismatches detected; skipping {skipped} keys and loading compatible subset.")
                model.load_state_dict(filtered, strict=False)
        else:
            try:
                model.load_state_dict(checkpoint, strict=False)
            except RuntimeError as e:
                current_sd = model.state_dict()
                filtered = {
                    k: v
                    for k, v in checkpoint.items()
                    if k in current_sd and isinstance(v, torch.Tensor) and v.shape == current_sd[k].shape
                }
                skipped = len(checkpoint) - len(filtered)
                print(f"[weights] load_state_dict shape mismatches detected; skipping {skipped} keys and loading compatible subset.")
                model.load_state_dict(filtered, strict=False)
    elif args.model_weights is None and args.mode in ["evaluation", "single"]:
        raise Exception('Model weights need to be provided for running inference.')
    
    return model, image_size 

def evaluate(
        model, 
        image_size, 
        args: argparse.Namespace
        ) -> None:
    """
    Evaluates the model on the SPEED dataset.
    OR 
    Performs a single inference on a single image and prints the results.

    Args:
        model: The model to evaluate
        image_size: The size of the input images (height, width)
        args: Command-line arguments containing evaluation parameters

    Returns:
        None
    """

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    keypoint_results = {}
    # If evaluating on the dataset, set up the dataloader and metrics, then evaluate
    
    if args.mode == "evaluation":
        # Set augmentation flags (not used during evaluation)
        args.no_pixel_augmentation = True
        args.no_spatial_augmentation = True

        #DATASET
        test_dataset = SPEEDDataset(
            dataset_root_dir=args.dataset_root_dir,
            split="test",
            img_size=image_size,
            bbox_json_path=args.bbox_json_path,
            args=args,
        )
        #DATALOADER
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        #METRICS, Initialize the metrics accumulators
        metric_keys = [
            "translation_metric",
            "rotation_metric",
            "total_metric",
            "relative_translation_metric",
            "relative_total_metric",
        ]
        avg_metrics = {k: [] for k in metric_keys}
        total_metrics = {k: [] for k in metric_keys}
        results = {}
        worst_sample_metric = -1.0
        worst_sample_idx = -1
        # EVALUATE
        sample_idx = 0
        with torch.no_grad():
            for batch in tqdm(test_loader):
                #Extract from the dictionary annotation
                images = batch['image']
                K = batch['K'] 
                translations = batch['translation']
                rotations = batch['rotation']
                bbox = batch['bbox']
                pts2d = batch['pts2d']
                pts3d = batch['pts3d']
                kpt_ops_values = batch['kpt_ops']

                # Move data to device
                images = images.to(device)
                translations = translations.to(device)
                rotations = rotations.to(device)

                # Forward pass
                pred_kpts = model(images)
                pred_kpts = restore_keypoints_batch(
                        kpts_ops=kpt_ops_values.to(pred_kpts.device), 
                        kpts_batch=pred_kpts
                        )
                preds_PnP = perform_PnP(
                    pts2d = pred_kpts.detach().cpu().numpy().astype(np.float32),
                    pts3d = pts3d.detach().cpu().numpy().astype(np.float32),
                    intrinsics = K.detach().cpu().numpy().astype(np.float32)
                )
                pred_translations = preds_PnP['pred_translations'].to(device).float()
                pred_rotations = preds_PnP['pred_rotations'].to(device).float()
                pred_rotations = rotation_matrix_to_quaternion(pred_rotations)

                # Convert relative translations to absolute translations if bbox is not None
                if bbox is not None and not torch.all(bbox.eq(0)):
                    # Send bbox to device
                    bbox = bbox.to(device)

                    # Convert bbox relative translations to absolute translations
                    translations = batched_bbox_relative_translation_to_translation(
                        translations, 
                        bbox, 
                        test_dataset.camera,  # type: ignore
                        args.no_translation_compensation
                    )

                # Convert rotation to centered rotation
                rotations = batched_get_absolute_orientation(
                    translation=translations,
                    rotation_quat=rotations,
                    no_rotation_compensation=args.no_rotation_compensation,
                )
                pred_rotations = batched_get_absolute_orientation(
                    translation=pred_translations,
                    rotation_quat=pred_rotations,
                    no_rotation_compensation=args.no_rotation_compensation,
                )

                # Compute batch metrics
                metrics = compute_metrics(pred_translations, translations, pred_rotations, rotations)
                if metrics["relative_total_metric"] > worst_sample_metric:
                    worst_sample_metric = metrics["relative_total_metric"]
                    worst_sample_idx = sample_idx

                # Accumulate batch metrics into total metrics
                total_metrics["translation_metric"].append(metrics["translation_metric_mean"])
                total_metrics["rotation_metric"].append(metrics["rotation_metric_mean"])
                total_metrics["total_metric"].append(metrics["total_metric"])
                total_metrics["relative_translation_metric"].append(metrics["relative_translation_metric_mean"])
                total_metrics["relative_total_metric"].append(metrics["relative_total_metric"])

                # Store predictions and ground truth
                for i in range(len(translations)):
                    results[str(sample_idx)] = {
                        "prediction": {
                            "translation": pred_translations[i].cpu().tolist(),
                            "rotation": pred_rotations[i].cpu().tolist(),
                        },
                        "ground_truth": {
                            "translation": translations[i].cpu().tolist(),
                            "rotation": rotations[i].cpu().tolist(),
                        },
                    }
                    keypoint_results[str(sample_idx)] = {
                        "prediction": pred_kpts[i].tolist(),
                        "ground_truth": pts2d[i].tolist()
                    }

                    sample_idx += 1
        
        # Compute the average metrics across the entire dataset
        avg_metrics = {k: sum(v) / len(v) for k, v in total_metrics.items()}

        # Add avg_metrics to results as a special key
        results["metrics"] = {
            # "translation_metric": float(avg_metrics["translation_metric"]),
            "rotation_metric": float(avg_metrics["rotation_metric"]),
            # "total_metric": float(avg_metrics["total_metric"]),
            "relative_translation_metric": float(avg_metrics["relative_translation_metric"]),
            "relative_total_metric": float(avg_metrics["relative_total_metric"]),
            "worst_relative_metric": float(worst_sample_metric),
            "worst_sample_index": str(worst_sample_idx),
        }

        # Print results
        print("Evaluation Results:")
        print(f"Average Translation Score: {avg_metrics['relative_translation_metric']:.4f}")
        print(f"Average Rotation Score: {avg_metrics['rotation_metric']:.4f}")
        print(f"Average Total Score: {avg_metrics['relative_total_metric']:.4f}")
        print(f"Worst metric: {worst_sample_metric} for image {worst_sample_idx}")

        # Save results to JSON if output path is provided
        if args.output_json:
            # Adjust the filename if faults are being injected
            output_json_path = args.output_json

            with open(output_json_path, "w") as f:
                json.dump(results, f, indent=4)

    elif args.mode == "single":
        img = np.array(Image.open(args.single_image_path).convert("RGB")) # type: ignore
        # grayscale to 3 channel
        if img.ndim == 2:
            img = np.expand_dims(img, 2)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        img.resize_(3, image_size[0], image_size[1]) # CNN expects a certain input size (384,384)
        img = img.to(device).unsqueeze(0).float() # [B,3,H,W]
        pts3d = SPEED_pts3d.unsqueeze(0)  # [B, 8, 3]
        camera = SPEED_Camera()
        K = camera.K
        with torch.no_grad():
            pred_kpts = model(img)
            pred_PnP = perform_PnP(
                    pts2d = pred_kpts.detach().cpu().numpy().astype(np.float32),
                    pts3d = pts3d.detach().numpy().astype(np.float32),
                    intrinsics = K.detach().numpy().astype(np.float32) # type: ignore
                )
            
        predicted_q = rotation_matrix_to_quaternion(pred_PnP['pred_rotations'][0])
        print("Single Image Inference Pose")
        print("Note the prediction is poor because the model was trained on cropped images.")
        print("The debug prediction is the model output for the single image.")
        with open(args.single_gt_path, "r") as f:
            gt = json.load(f)
        print(f"Predicted Translation: {pred_PnP['pred_translations'][0].numpy()}")
        print(f"Debug Translation: {gt[0]['debug_r']}")
        print(f"Ground Truth Translation: {gt[0]['r_Vo2To_vbs_true']}")
        print(f"Predicted Rotation: {predicted_q}")
        print(f"Debug Rotation: {gt[0]['debug_q']}")
        print(f"Ground Truth Rotation: {gt[0]['q_vbs2tango']}")
        
        
if __name__ == "__main__":
    parser = process_commandline_inputs(return_parser=True)
    args = parser.parse_args()

    model, image_size = obtain_damoL45_kpt(args=args)
    evaluate(
        model=model, 
        image_size=image_size, 
        args=args
    )

# Example evaluate on SPEED in bash
# Set the paths yourself and run the following command in the terminal:
"""
DATASET=/mnt/cvlab/scratch/cvlab/home/price/FI/SimplePose/SPEED_FIXED/
BBOX_JSON_PATH=$DATASET/all_annotations.json
MODEL_WEIGHTS=/mnt/cvlab/scratch/cvlab/home/price/FI/SimplePose/experiments/base_models/damo_L45_L/keypoint/model.pth
RESULTS=/mnt/cvlab/scratch/cvlab/home/price/FI/SimplePose/for_deployment/output/results.json
LOG_DIR=$RESULTS

python3 eval.py \
    --mode evaluation \
    --model_weights $MODEL_WEIGHTS \
    --dataset_root_dir $DATASET/ \
    --bbox_json_path $BBOX_JSON_PATH \
    --output_json $RESULTS
"""

# Example single image inference in bash
# Set the paths yourself and run the following command in the terminal:
"""
SINGLE_IMAGE_DIR=/mnt/cvlab/scratch/cvlab/home/price/FI/SimplePose/for_deployment/single_image/
MODEL_WEIGHTS=/mnt/cvlab/scratch/cvlab/home/price/FI/SimplePose/experiments/base_models/damo_L45_L/keypoint/model.pth

python3 eval.py \
    --mode single \
    --model_weights $MODEL_WEIGHTS \
    --single_image_path $SINGLE_IMAGE_DIR/img000004.jpg \
    --single_gt_path $SINGLE_IMAGE_DIR/img000004_gt.json
"""