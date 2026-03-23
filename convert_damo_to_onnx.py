#!/usr/bin/env python3
"""
Convert DamoPose (CNN-based) model to ONNX format for deployment.
This creates a portable, optimized model file for edge GPU deployment.
"""

import argparse
import os
import sys

import torch
import onnx

from src.models import DamoPose, SUPPORTED_CNN_MODELS, CNN_MODELS


def convert_damo_to_onnx(args: argparse.Namespace) -> None:
    """
    Convert the DamoPose model to ONNX format.

    Args:
        args (argparse.Namespace): Command line arguments
    """
    
    # Verify output directory
    os.makedirs(os.path.dirname(args.onnx_model_path) or ".", exist_ok=True)
    
    print(f"[INFO] Loading DamoPose model: {args.model_backbone}")
    # Resolve whether we are using direct regression or a PnP/keypoint head
    resolved_direct_regression = getattr(args, "direct_regression", None)
    if resolved_direct_regression is None:
        resolved_direct_regression = not getattr(args, "use_pnp", False)

    print(f"[INFO] Configuration:")
    print(f"  - num_hidden_layers: {args.num_hidden_layers}")
    print(f"  - hidden_layer_dim: {args.hidden_layer_dim}")
    print(f"  - rotation_format: {args.rotation_format}")
    print(f"  - direct_regression: {resolved_direct_regression}")
    
    # Set up output dimensions
    out_dim_rotation = 4 if args.rotation_format == "quaternion" else 6
    out_dim_translation = 3
    
    # Check if weights file exists
    if not os.path.exists(args.model_weights):
        raise FileNotFoundError(f"Model weights not found: {args.model_weights}")
    
    # Create the model
    model = DamoPose(
        cnn_model=args.model_backbone,
        direct_regression=resolved_direct_regression,
        num_hidden_layers=args.num_hidden_layers,
        hidden_layer_dim=args.hidden_layer_dim,
        out_dim_translation=out_dim_translation,
        out_dim_rotation=out_dim_rotation,
        merge_outputs=args.merge_outputs,
        damo_load_verbose=False,
        damo_load_strict=False,
        damo_min_load_ratio=0.6,
    )
    
    # Load weights
    print(f"[INFO] Loading model weights from: {args.model_weights}")
    checkpoint = torch.load(args.model_weights, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()
    
    # Get input image size from CNN model configuration
    image_size = CNN_MODELS[args.model_backbone][3]  # (height, width)
    print(f"[INFO] Input image size: {image_size}")
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, *image_size)
    
    # Export to ONNX
    print(f"[INFO] Exporting to ONNX...")
    # Decide ONNX output names based on head type
    if not resolved_direct_regression:
        output_names = ["keypoints"]
    else:
        output_names = ["output"] if args.merge_outputs else ["translation", "rotation"]
    
    torch.onnx.export(
        model,
        (dummy_input,),
        args.onnx_model_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["image"],
        output_names=output_names,
        verbose=False,
    )
    
    # Verify the exported model
    print(f"[INFO] Verifying ONNX model...")
    onnx_model = onnx.load(args.onnx_model_path)
    onnx.checker.check_model(onnx_model)
    
    print(f"[SUCCESS] Model successfully exported to: {args.onnx_model_path}")
    print(f"[INFO] Model outputs: {output_names}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert DamoPose CNN model to ONNX for edge deployment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--model_backbone",
        type=str,
        choices=SUPPORTED_CNN_MODELS,
        default="damo_L45_L",
        help="CNN backbone model to use",
    )
    parser.add_argument(
        "--model_weights",
        type=str,
        required=True,
        help="Path to model weights (.pth file)",
    )
    parser.add_argument(
        "--onnx_model_path",
        type=str,
        default="damo_model.onnx",
        help="Output path for ONNX model",
    )
    parser.add_argument(
        "--num_hidden_layers",
        type=int,
        default=3,
        help="Number of hidden layers in regression heads",
    )
    parser.add_argument(
        "--hidden_layer_dim",
        type=int,
        default=800,
        help="Dimension of hidden layers",
    )
    parser.add_argument(
        "--rotation_format",
        type=str,
        choices=["quaternion", "matrix"],
        default="matrix",
        help="Rotation representation format",
    )
    parser.add_argument(
        "--use_pnp",
        action="store_true",
        default=False,
        help="Use PnP/keypoint head instead of direct regression",
    )
    parser.add_argument(
        "--merge_outputs",
        action="store_true",
        default=False,
        help="Merge translation and rotation outputs",
    )
    
    args = parser.parse_args()
    convert_damo_to_onnx(args)
