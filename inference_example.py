"""
Example inference script for SimplePose DamoPose model on edge device.
This demonstrates how to use the exported ONNX model for inference.
"""

import onnxruntime as rt
import numpy as np
import cv2
from pathlib import Path


class SimplePoseInference:
    """Wrapper for SimplePose ONNX model inference"""
    
    def __init__(self, model_path: str, use_gpu: bool = True):
        """
        Initialize inference session.
        
        Args:
            model_path: Path to ONNX model
            use_gpu: Use GPU if available (Jetson, NVIDIA GPU)
        """
        providers = []
        if use_gpu:
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        
        self.session = rt.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        
        print(f"[INFO] Model loaded on device: {self.session.get_providers()}")
        print(f"[INFO] Input: {self.input_name}")
        print(f"[INFO] Outputs: {self.output_names}")
    
    def preprocess(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess image for inference.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Preprocessed image tensor (1, 3, 384, 384)
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to 384x384
        image = cv2.resize(image, (384, 384))
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        
        # Convert to batch tensor (1, 3, H, W)
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)
        
        return image.astype(np.float32)
    
    def infer(self, image_tensor: np.ndarray) -> dict:
        """
        Run inference.
        
        Args:
            image_tensor: Preprocessed image tensor
            
        Returns:
            Dictionary with translation and rotation outputs
        """
        outputs = self.session.run(self.output_names, {self.input_name: image_tensor})
        
        result = {}
        for name, output in zip(self.output_names, outputs):
            result[name] = output
        
        return result
    
    def rot6d_to_rotmat(self, rot6d: np.ndarray) -> np.ndarray:
        """
        Convert 6D rotation representation to 3x3 rotation matrix.
        
        Args:
            rot6d: (batch, 6) 6D rotation
            
        Returns:
            (batch, 3, 3) rotation matrices
        """
        batch_size = rot6d.shape[0]
        rot6d = rot6d.reshape(batch_size, 3, 2)
        
        # Gram-Schmidt orthogonalization
        x = rot6d[:, :, 0]
        y = rot6d[:, :, 1]
        
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
        y = y - (x * y).sum(axis=1, keepdims=True) * x
        y = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-8)
        
        z = np.cross(x, y, axis=1)
        
        return np.stack([x, y, z], axis=2)
    
    def process_image(self, image_path: str) -> dict:
        """
        Complete pipeline: load, preprocess, infer, postprocess.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Dictionary with pose results
        """
        # Preprocess
        image_tensor = self.preprocess(image_path)
        
        # Inference
        outputs = self.infer(image_tensor)
        
        # Postprocess
        translation = outputs['translation'][0]  # (3,)
        rotation_6d = outputs['rotation'][0]  # (6,)
        rotation_mat = self.rot6d_to_rotmat(rotation_6d[np.newaxis, :])[0]  # (3, 3)
        
        return {
            'translation': translation,
            'rotation_6d': rotation_6d,
            'rotation_matrix': rotation_mat,
        }


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SimplePose ONNX inference example")
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    args = parser.parse_args()
    
    # Initialize
    inference = SimplePoseInference(args.model, use_gpu=args.gpu)
    
    # Run inference
    print(f"\n[INFO] Processing image: {args.image}")
    results = inference.process_image(args.image)
    
    # Display results
    print(f"\n[RESULTS]")
    print(f"Translation (XYZ): {results['translation']}")
    print(f"Rotation (6D): {results['rotation_6d']}")
    print(f"Rotation Matrix:\n{results['rotation_matrix']}")


if __name__ == "__main__":
    main()
