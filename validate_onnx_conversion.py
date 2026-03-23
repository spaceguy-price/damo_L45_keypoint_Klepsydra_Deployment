#!/usr/bin/env python3
"""
Validate ONNX model conversion by comparing outputs with PyTorch model.
Tests accuracy of the ONNX export against the original PyTorch model.
"""

import argparse
import os
import json
from pathlib import Path

import torch
import numpy as np
import onnxruntime as rt
from tqdm import tqdm

from src.models import DamoPose, SUPPORTED_CNN_MODELS, CNN_MODELS
from src.datasets import SPEEDDataset


class OMNXValidator:
    """Validate ONNX model against PyTorch reference"""
    
    def __init__(self, torch_model_path: str, onnx_model_path: str, 
                 model_backbone: str, num_hidden_layers: int, hidden_layer_dim: int,
                 rotation_format: str = "matrix"):
        """
        Initialize validator with both models.
        
        Args:
            torch_model_path: Path to .pth weights
            onnx_model_path: Path to .onnx model
            model_backbone: CNN backbone name
            num_hidden_layers: Number of hidden layers
            hidden_layer_dim: Hidden layer dimension
            rotation_format: Rotation representation
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Using device: {self.device}")
        
        out_dim_rotation = 4 if rotation_format == "quaternion" else 6
        
        # Load PyTorch model
        print(f"[INFO] Loading PyTorch model...")
        self.torch_model = DamoPose(
            cnn_model=model_backbone,
            direct_regression=False,
            num_hidden_layers=num_hidden_layers,
            hidden_layer_dim=hidden_layer_dim,
            out_dim_translation=3,
            out_dim_rotation=out_dim_rotation,
        ).to(self.device)
        
        checkpoint = torch.load(torch_model_path, map_location=self.device, weights_only=True)
        self.torch_model.load_state_dict(checkpoint)
        self.torch_model.eval()
        
        # Load ONNX model
        print(f"[INFO] Loading ONNX model...")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == "cuda" else ['CPUExecutionProvider']
        self.onnx_session = rt.InferenceSession(onnx_model_path, providers=providers)
        self.input_name = self.onnx_session.get_inputs()[0].name
        
        # Get image size
        self.image_size = CNN_MODELS[model_backbone][3]
        print(f"[INFO] Image size: {self.image_size}")
    
    def compare_outputs(self, image_batch: torch.Tensor) -> dict:
        """
        Compare PyTorch and ONNX outputs on a batch.
        
        Args:
            image_batch: (B, 3, H, W) image tensor
            
        Returns:
            Dictionary with comparison metrics
        """
        with torch.no_grad():
            # PyTorch inference
            torch_out = self.torch_model(image_batch.to(self.device))
            if isinstance(torch_out, tuple):
                torch_translation, torch_rotation = torch_out
            else:
                torch_translation = torch_out[:, :3]
                torch_rotation = torch_out[:, 3:]
        
        torch_translation = torch_translation.cpu().numpy()
        torch_rotation = torch_rotation.cpu().numpy()
        
        # ONNX inference
        image_np = image_batch.cpu().numpy().astype(np.float32)
        onnx_outputs = self.onnx_session.run(None, {self.input_name: image_np})
        onnx_translation = onnx_outputs[0]  # First output
        onnx_rotation = onnx_outputs[1]     # Second output
        
        # Compute differences
        trans_diff = np.abs(torch_translation - onnx_translation)
        rot_diff = np.abs(torch_rotation - onnx_rotation)
        
        return {
            'trans_mean_diff': float(np.mean(trans_diff)),
            'trans_max_diff': float(np.max(trans_diff)),
            'trans_std_diff': float(np.std(trans_diff)),
            'rot_mean_diff': float(np.mean(rot_diff)),
            'rot_max_diff': float(np.max(rot_diff)),
            'rot_std_diff': float(np.std(rot_diff)),
            'torch_trans': torch_translation.tolist(),
            'onnx_trans': onnx_translation.tolist(),
            'torch_rot': torch_rotation.tolist(),
            'onnx_rot': onnx_rotation.tolist(),
        }
    
    def validate_on_dataset(self, dataset: SPEEDDataset, num_samples: int = 50) -> dict:
        """
        Validate on dataset samples.
        
        Args:
            dataset: SPEED dataset
            num_samples: Number of samples to test
            
        Returns:
            Aggregated validation metrics
        """
        print(f"\n[INFO] Validating ONNX on {num_samples} dataset samples...")
        
        all_trans_diffs = []
        all_rot_diffs = []
        
        for i in tqdm(range(min(num_samples, len(dataset))), desc="Validating"):
            sample = dataset[i]
            image = sample['image'].unsqueeze(0)  # Add batch dim
            
            results = self.compare_outputs(image)
            all_trans_diffs.append(results['trans_mean_diff'])
            all_rot_diffs.append(results['rot_mean_diff'])
        
        all_trans_diffs = np.array(all_trans_diffs)
        all_rot_diffs = np.array(all_rot_diffs)
        
        return {
            'num_samples': num_samples,
            'translation': {
                'mean_diff': float(np.mean(all_trans_diffs)),
                'max_diff': float(np.max(all_trans_diffs)),
                'std_diff': float(np.std(all_trans_diffs)),
            },
            'rotation': {
                'mean_diff': float(np.mean(all_rot_diffs)),
                'max_diff': float(np.max(all_rot_diffs)),
                'std_diff': float(np.std(all_rot_diffs)),
            }
        }


def main():
    parser = argparse.ArgumentParser(
        description="Validate ONNX model conversion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--torch_model_weights",
        type=str,
        required=True,
        help="Path to PyTorch model weights (.pth)",
    )
    parser.add_argument(
        "--onnx_model_path",
        type=str,
        required=True,
        help="Path to ONNX model (.onnx)",
    )
    parser.add_argument(
        "--dataset_root_dir",
        type=str,
        required=True,
        help="Root directory of SPEED dataset",
    )
    parser.add_argument(
        "--bbox_json_path",
        type=str,
        required=True,
        help="Path to bbox JSON",
    )
    parser.add_argument(
        "--model_backbone",
        type=str,
        choices=SUPPORTED_CNN_MODELS,
        default="damo_L45_L",
        help="CNN backbone",
    )
    parser.add_argument(
        "--num_hidden_layers",
        type=int,
        default=3,
        help="Number of hidden layers",
    )
    parser.add_argument(
        "--hidden_layer_dim",
        type=int,
        default=800,
        help="Hidden layer dimension",
    )
    parser.add_argument(
        "--rotation_format",
        type=str,
        choices=["quaternion", "matrix"],
        default="matrix",
        help="Rotation format",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Number of dataset samples to validate",
    )
    parser.add_argument(
        "--output_report",
        type=str,
        default="onnx_validation_report.json",
        help="Path to save validation report",
    )
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = OMNXValidator(
        torch_model_path=args.torch_model_weights,
        onnx_model_path=args.onnx_model_path,
        model_backbone=args.model_backbone,
        num_hidden_layers=args.num_hidden_layers,
        hidden_layer_dim=args.hidden_layer_dim,
        rotation_format=args.rotation_format,
    )
    
    # Load dataset
    print(f"[INFO] Loading dataset from {args.dataset_root_dir}")
    dataset = SPEEDDataset(
        dataset_root_dir=args.dataset_root_dir,
        split="test",
        img_size=(384, 384),
        bbox_json_path=args.bbox_json_path,
    )
    
    # Run validation
    report = validator.validate_on_dataset(dataset, num_samples=args.num_samples)
    
    # Print and save results
    print("\n" + "="*60)
    print("ONNX VALIDATION REPORT")
    print("="*60)
    print(f"Samples: {report['num_samples']}")
    print(f"\nTranslation Error:")
    print(f"  Mean diff: {report['translation']['mean_diff']:.6f}")
    print(f"  Max diff:  {report['translation']['max_diff']:.6f}")
    print(f"  Std diff:  {report['translation']['std_diff']:.6f}")
    print(f"\nRotation Error:")
    print(f"  Mean diff: {report['rotation']['mean_diff']:.6f}")
    print(f"  Max diff:  {report['rotation']['max_diff']:.6f}")
    print(f"  Std diff:  {report['rotation']['std_diff']:.6f}")
    print("="*60)
    
    # Determine if conversion is valid
    trans_threshold = 1e-4
    rot_threshold = 1e-4
    
    trans_valid = report['translation']['max_diff'] < trans_threshold
    rot_valid = report['rotation']['max_diff'] < rot_threshold
    
    if trans_valid and rot_valid:
        print("\n✓ ONNX conversion VALID - outputs match PyTorch model")
    else:
        print("\n✗ ONNX conversion may have issues:")
        if not trans_valid:
            print(f"  - Translation error too high ({report['translation']['max_diff']:.6f})")
        if not rot_valid:
            print(f"  - Rotation error too high ({report['rotation']['max_diff']:.6f})")
    
    # Save report
    with open(args.output_report, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n[INFO] Report saved to: {args.output_report}")


if __name__ == "__main__":
    main()
