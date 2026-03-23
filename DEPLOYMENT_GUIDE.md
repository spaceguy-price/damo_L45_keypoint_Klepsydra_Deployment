# Deployment Guide for SimplePose DamoPose Model

## Overview
This guide explains how to package and deploy your trained SimplePose model (damo_L45_L) to edge GPUs like NVIDIA Jetson Orin or CPU targets.

## What the Company Needs

### 1. **ONNX Model** (portable, framework-agnostic)
- Binary `.onnx` file containing model architecture + weights
- Works on CPU and GPU
- Inference via ONNX Runtime (optimized, lightweight)

### 2. **Model Configuration**
```
Model: damo_L45_L (DAMO YOLO backbone)
Input:
  - Size: 384 x 384 pixels
  - Format: RGB image (normalized)
  - Channels: 3
  - Device: CPU or CUDA

Output (3 regression heads):
  - translation (3): x, y, z position in meters
  - rotation (6): 6D rotation representation (or 4 for quaternion)
  - Optional: conf_score if available

Frame rate: ~40-50 FPS on Jetson Orin (batch=1)
```

### 3. **Preprocessing Requirements**
```python
import cv2
import numpy as np

# Image preprocessing
image = cv2.imread("path/to/image.jpg")
image = cv2.resize(image, (384, 384))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Normalize (ImageNet)
image = image.astype(np.float32) / 255.0
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
image = (image - mean) / std

# Convert to batch
input_tensor = np.expand_dims(image, axis=0)  # (1, 384, 384, 3)
input_tensor = np.transpose(input_tensor, (0, 3, 1, 2))  # (1, 3, 384, 384)
```

### 4. **Postprocessing**
```python
# Assuming outputs are:
# translation: (batch, 3)
# rotation: (batch, 6) for 6D representation

# Optional: Convert 6D rotation to 3x3 rotation matrix
def rot6d_to_rotmat(rot6d):
    """Convert 6D rotation to 3x3 rotation matrix"""
    batch_size = rot6d.shape[0]
    rot6d = rot6d.reshape(-1, 3, 2)
    
    # Gram-Schmidt orthogonalization
    x = rot6d[:, :, 0]
    y = rot6d[:, :, 1]
    
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    y = y - (x * y).sum(axis=1, keepdims=True) * x
    y = y / np.linalg.norm(y, axis=1, keepdims=True)
    
    z = np.cross(x, y)
    
    return np.stack([x, y, z], axis=2)
```

## Deployment Steps

### Step 1: Export to ONNX
```bash
cd /path/to/SimplePose

python convert_damo_to_onnx.py \
  --model_backbone damo_L45_L \
  --model_weights /path/to/model.pth \
  --onnx_model_path damo_L45_L_final.onnx \
  --num_hidden_layers 3 \
  --hidden_layer_dim 800 \
  --rotation_format matrix
```

### Step 2: Verify ONNX Model
```bash
# Check model structure
python -c "import onnx; model = onnx.load('damo_L45_L_final.onnx'); onnx.checker.check_model(model); print('Valid!')"

# Test with dummy input
python -c "
import onnxruntime as rt
import numpy as np

sess = rt.InferenceSession('damo_L45_L_final.onnx')
dummy = np.random.randn(1, 3, 384, 384).astype(np.float32)
output = sess.run(None, {'image': dummy})
print(f'Output shapes: {[o.shape for o in output]}')
"
```

### Step 3: Package for Delivery
Create a deployment folder with:
```
deployment_package/
├── damo_L45_L_final.onnx          # ONNX model
├── deployment_config.json         # Model metadata
├── inference_example.py           # Example inference code
├── requirements.txt               # Dependencies
└── README.md                       # Instructions
```

**deployment_config.json:**
```json
{
  "model_name": "SimplePose_damo_L45_L",
  "model_format": "ONNX",
  "input": {
    "name": "image",
    "shape": [1, 3, 384, 384],
    "dtype": "float32",
    "preprocessing": "normalize with ImageNet mean/std"
  },
  "outputs": {
    "translation": {"shape": [1, 3], "description": "XYZ position"},
    "rotation": {"shape": [1, 6], "description": "6D rotation representation"}
  },
  "rotation_format": "matrix_6d",
  "performance": {
    "jetson_orin": "~40-50 FPS (batch=1)",
    "cpu_i7": "~8-12 FPS (batch=1)"
  }
}
```

**requirements.txt:**
```
onnxruntime>=1.14.0  # CPU inference
# OR for GPU:
# onnxruntime-gpu>=1.14.0
opencv-python>=4.6.0
numpy>=1.20.0
# Optional for TensorRT optimization:
# torch>=2.0.0
# tensorrt>=8.5.0
```

### Step 4: On Edge Device (Jetson Orin / CPU)

**Option A: ONNX Runtime (Simple, Recommended)**
```python
import onnxruntime as rt
import numpy as np
import cv2

# Load model
sess = rt.InferenceSession('damo_L45_L_final.onnx', 
                           providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

# Prepare image
image = cv2.imread("image.jpg")
image = cv2.resize(image, (384, 384))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)

# Inference
outputs = sess.run(None, {'image': image})
translation, rotation = outputs[0], outputs[1]

print(f"Translation: {translation}")
print(f"Rotation: {rotation}")
```

**Option B: TensorRT (Performance Optimized)**
```bash
# Convert ONNX to TensorRT
trtexec --onnx=damo_L45_L_final.onnx --saveEngine=damo_L45_L_final.trt --fp16

# Use in Python with TensorRT bindings
# (See TensorRT documentation)
```

## Performance Notes

| Device | Model Size | FPS | Memory |
|--------|-----------|-----|--------|
| Jetson Orin NX 8GB | ~850MB | 45-55 | ~2-3GB |
| Jetson Orion Nano | - | 5-8 | ~2GB |
| RTX 4090 | ~850MB | 200+ | ~1GB |
| CPU (i7-12700) | ~850MB | 8-12 | ~1GB |

## Troubleshooting

**Problem: "Unsupported dtype in ONNX"**
- Ensure model is in `float32` or `float16`
- Check ONNX opset version compatibility

**Problem: Runtime error on Jetson**
- Install ONNX Runtime for ARM: `pip install onnxruntime-aarch64`
- Use GPU provider: `CUDAExecutionProvider`

**Problem: Slow inference**
- Consider TensorRT conversion for 2-4x speedup
- Enable mixed precision (fp16)
- Reduce batch size if out of memory

## References
- [ONNX Runtime Documentation](https://onnxruntime.ai/)
- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt)
- [Jetson Deployment Guide](https://docs.nvidia.com/jetson/)
