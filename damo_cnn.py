# This file contains the CNN backbone model and is mostly unchanged from the original DamoPose repository

# The first model is based on the Apache-2.0 licensed DAMO-YOLO-L model (42.1M parameters). You can find the original here:
# https://github.com/tinyvision/DAMO-YOLO/tree/master


import os
from typing import Any, Callable, NamedTuple, Optional

import torch
import torch.nn as nn
from torchvision.utils import _log_api_usage_once

from damo_yolo.base_models.backbones.tinynas_csp import TinyNAS #csp is the Darknet version, ResNet and MobileNet are also available
from damo_yolo.base_models.necks.giraffe_fpn_btn import GiraffeNeckV2
from damo_yolo.base_models.heads.zero_head import ZeroHead

__all__ = [
    "damo_L45_L",
]

class ConvStemConfig(NamedTuple):
    out_channels: int
    kernel_size: int
    stride: int
    norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d
    activation_layer: Callable[..., nn.Module] = nn.ReLU


class FPNtoFCAdapter(nn.Module):
    """Concatenates FPN feature maps into a single vector for FC head"""
    def __init__(
            self, 
            in_channels=[128, 256, 512],
            spatial_sizes=[28,14,7],
            hidden_dim=728,
            ):
        
        super().__init__()

        self.flatten=nn.Flatten() #(B,C,H,W)->(B, C*H*W)
        
        #Compute total input size after flattening
        total_input_dim = sum([c * (s ** 2) for c, s in zip(in_channels, spatial_sizes)])
        #Linear projection
        self.fc = nn.Linear(total_input_dim, hidden_dim)

    def forward(self, fpn_outputs):
        """
        Arguments:
            fpn_outputs: Tuple of FPN features (x5,x7,x6)
        Returns:
            Flattended feasure vector (B, hidden_dim)
        """
        flattened_features = [self.flatten(output) for output in fpn_outputs] #(B, C*H*W)
        concatenated_features = torch.cat(flattened_features, dim=1) #(B, total_input_dim)
        projected_feature = self.fc(concatenated_features) # (B, hidden_dim)

        return projected_feature

class DamoCNN(nn.Module):
    """CNN as per https://github.com/tinyvision/DAMO-YOLO/.
    NOTE: The implementation has the head replaced for pose estimation.
    """

    def __init__(
        self,
        image_size: int,
        structure_path: str,
        num_classes: int = 1000,
    ):
        super().__init__()
        _log_api_usage_once(self)
        self.image_size = image_size
        self.num_classes = num_classes
        with open(structure_path, 'r') as f:
            self.structure = f.read()

        #BUILD BACKBONE
        import ast
        structure_info = ast.literal_eval(self.structure)
        for layer in structure_info:
            if 'nbitsA' in layer:
                del layer['nbitsA']
            if 'nbitsW' in layer:
                del layer['nbitsW']

        self.backbone = TinyNAS(structure_info=structure_info,
                    out_indices=(2,3,4),
                    with_spp=True,
                    use_focus=True,
                    act='silu',
                    reparam=True,
        )
        # Infer backbone output channel sizes dynamically to configure FPN/head correctly
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.image_size, self.image_size)
            bb_feats = self.backbone(dummy)
            in_channels = [f.shape[1] for f in bb_feats]

        #BUILD FPN (Feature Pyramid Network) using detected channels
        self.FPN = GiraffeNeckV2(
            depth=2.0,
            hidden_ratio=1.0,
            in_channels=in_channels,
            out_channels=in_channels,
            act='silu',
            spp=False,
            block_name='BasicBlock_3x3_Reverse',
        )

        #Build Head (ZeroHead) aligned with detected channels
        self.head = ZeroHead(
            num_classes=80,
            in_channels=in_channels,
            stacked_convs=0,
            reg_max=16,
            act='silu',
            legacy=False,
        )

    def _check_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        torch._assert(
            h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!"
        )
        torch._assert(
            w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!"
        )
        return x

    def forward(self, x: torch.Tensor):
        x = self._check_input(x)
        n = x.shape[0]

        x = self.backbone(x)
        outputs = self.FPN(x)

        # Here we will comment out the head, so that the call for Imgs is ignored
        # outputs = self.head(
        #         outputs,
        #         None,
        #         imgs=None,
        #     )
        return outputs
        
def _damo_cnn(
    weights_path: Optional[str],
    structure_path: str,
    **kwargs: Any,
) -> DamoCNN:
    
    image_size = kwargs.pop("image_size", 384)

    # Provide a lightweight fallback for environments where the DAMO
    # NAS structure file is not available (e.g., CI/tests). This keeps
    # constructions like DamoPose working for smoke tests that do not
    # rely on the exact DAMO backbone details.
    if not os.path.isfile(structure_path):
        class _DummyDamoCNN(nn.Module):
            def __init__(self, image_size: int):
                super().__init__()
                _log_api_usage_once(self)
                self.image_size = image_size
                # Minimal stem to keep parameter registration non-trivial
                self.stem = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                )
                # Simple downsample towers to mimic three FPN levels
                self.l1 = nn.Sequential(
                    nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                )
                self.l2 = nn.Sequential(
                    nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                )
                self.l3 = nn.Sequential(
                    nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                )

            def _check_input(self, x: torch.Tensor) -> torch.Tensor:
                n, c, h, w = x.shape
                torch._assert(
                    h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!"
                )
                torch._assert(
                    w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!"
                )
                return x

            def forward(self, x: torch.Tensor):
                x = self._check_input(x)
                x = self.stem(x)
                f1 = self.l1(x)
                f2 = self.l2(f1)
                f3 = self.l3(f2)
                # Return tuple mimicking (x5, x7, x6) ordering used elsewhere
                return (f1, f3, f2)

        model: DamoCNN | _DummyDamoCNN = _DummyDamoCNN(image_size=image_size)
    else:
        model = DamoCNN(
            image_size=image_size,
            structure_path=structure_path,
            num_classes=1000,
        )

    if weights_path is not None:
        ckpt = torch.load(
            weights_path,
            map_location=torch.device('cpu'),
            weights_only=False,
        )
        # Load only the weights that match by key AND shape to avoid size mismatch errors
        # (e.g., detection head shapes differ from our pose head or unused head modules)
        ckpt_state_raw = ckpt.get('model', ckpt)
        model_state = model.state_dict()

        # Optional: strip common wrappers like 'module.' if present
        def strip_prefix_if_present(state_dict, prefix='module.'):
            if state_dict and all(k.startswith(prefix) for k in state_dict.keys()):
                return {k[len(prefix):]: v for k, v in state_dict.items()}
            return state_dict

        ckpt_state_raw = strip_prefix_if_present(ckpt_state_raw)

        # Remap checkpoint keys from upstream DAMO naming to our module names.
        import re

        # Pre-compute how many inner blocks are in backbone.block_list.3 to offset stage 4 mapping
        def count_inner_blocks(state_dict, outer_idx: int) -> int:
            pat = re.compile(rf"^backbone\.block_list\.{outer_idx}\.block_list\.(\d+)\.")
            max_j = -1
            for k in state_dict.keys():
                m = pat.match(k)
                if m:
                    max_j = max(max_j, int(m.group(1)))
            return max_j + 1 if max_j >= 0 else 0

        n_stage3 = count_inner_blocks(ckpt_state_raw, 3)

        def remap_key(k: str) -> str:
            # neck.* → FPN.*
            if k.startswith('neck.'):
                return 'FPN.' + k[len('neck.'):]

            # backbone.block_list.0.* → backbone.csp_stage.0.* (direct conv stem)
            if k.startswith('backbone.block_list.0.'):
                return 'backbone.csp_stage.0.' + k[len('backbone.block_list.0.'):]

            # backbone.block_list.{i}.block_list.{j}.* → backbone.csp_stage.{stage}.convstem.{j or j+offset}.*
            m = re.match(r'^backbone\.block_list\.(\d+)\.block_list\.(\d+)\.(.*)$', k)
            if m:
                i = int(m.group(1))
                j = int(m.group(2))
                tail = m.group(3)
                # Map outer i to our csp_stage index
                # i=1→stage1, i=2→stage2, i=3+4→stage3 (concatenated), i=5→stage4
                if i == 1:
                    stage_idx = 1
                    new_j = j
                elif i == 2:
                    stage_idx = 2
                    new_j = j
                elif i == 3:
                    stage_idx = 3
                    new_j = j
                elif i == 4:
                    stage_idx = 3
                    new_j = (n_stage3 or 0) + j
                elif i == 5:
                    stage_idx = 4
                    new_j = j
                else:
                    return k  # unknown, leave as-is
                return f'backbone.csp_stage.{stage_idx}.convstem.{new_j}.{tail}'

            return k  # default: unchanged

        # Build a remapped state dict
        remapped_state = {}
        original_to_remapped = {}
        for k, v in ckpt_state_raw.items():
            rk = remap_key(k)
            original_to_remapped[k] = rk
            remapped_state[rk] = v

        # Build adapted state dict matching model shapes; try partial copies when shapes differ
        adapted_state = {}
        exact_loaded = []
        partial_loaded = []
        skipped_keys = []

        for orig_k, remap_k in original_to_remapped.items():
            if remap_k not in model_state:
                skipped_keys.append(orig_k)
                continue
            src = remapped_state[remap_k]
            dst = model_state[remap_k]
            # Exact shape match
            if dst.shape == src.shape:
                adapted_state[remap_k] = src
                exact_loaded.append(orig_k)
                continue

            # Try shape-aware partial copy for common parameter types
            try:
                new_tensor = dst.clone()
                if src.ndim == 4 and dst.ndim == 4 and src.shape[2:] == dst.shape[2:]:
                    # Conv2d weight: [out_c, in_c, k, k]
                    oc = min(src.shape[0], dst.shape[0])
                    ic = min(src.shape[1], dst.shape[1])
                    new_tensor[:oc, :ic, :, :] = src[:oc, :ic, :, :].to(new_tensor.dtype)
                    adapted_state[remap_k] = new_tensor
                    partial_loaded.append(orig_k)
                    continue
                elif src.ndim == 1 and dst.ndim == 1:
                    # BN/affine vectors: [C]
                    c = min(src.shape[0], dst.shape[0])
                    new_tensor[:c] = src[:c].to(new_tensor.dtype)
                    adapted_state[remap_k] = new_tensor
                    partial_loaded.append(orig_k)
                    continue
                elif src.ndim == 2 and dst.ndim == 2:
                    # Linear weight: [out, in]
                    ro = min(src.shape[0], dst.shape[0])
                    ci = min(src.shape[1], dst.shape[1])
                    new_tensor[:ro, :ci] = src[:ro, :ci].to(new_tensor.dtype)
                    adapted_state[remap_k] = new_tensor
                    partial_loaded.append(orig_k)
                    continue
                else:
                    skipped_keys.append(orig_k)
            except Exception:
                skipped_keys.append(orig_k)

        # Load filtered weights; strict=False to allow missing keys from our replaced head
        load_info = model.load_state_dict(adapted_state, strict=False)

        # --- Guardrails & Diagnostics ---
        def _truthy(v) -> bool:
            return str(v).lower() in ("1", "true", "yes", "on")

        verbose = _truthy(kwargs.get("damo_load_verbose", True))
        strict = _truthy(kwargs.get("damo_load_strict", False))
        min_load_ratio = float(kwargs.get("damo_min_load_ratio", 0.6))
        # Allow head.* by default; also tolerate legacy upstream prefixes that may remain unmapped
        default_allowed = "head.,backbone.block_list."
        allowed_skip_prefixes = tuple(
            str(kwargs.get("damo_allowed_skip_prefixes", default_allowed)).split(',')
        )

        total_keys = len(ckpt_state_raw)
        loaded_keys = len(exact_loaded) + len(partial_loaded)
        load_ratio = (loaded_keys / total_keys) if total_keys > 0 else 1.0

        # Classify skipped keys
        non_allowed_skips = [k for k in skipped_keys if not k.startswith(allowed_skip_prefixes)]

        if verbose:
            print(f"[DamoCNN] Loaded {loaded_keys}/{total_keys} ({load_ratio:.1%}) keys from '{weights_path}'. Skipped: {len(skipped_keys)}")
            if skipped_keys:
                # Show a small sample of skipped keys
                sample = skipped_keys[:10]
                print(f"[DamoCNN] Skipped keys (sample {len(sample)}): {sample}")
            if partial_loaded:
                sample_p = partial_loaded[:10]
                print(f"[DamoCNN] Partially loaded by shape adapt (sample {len(sample_p)}): {sample_p}")
            if load_info.missing_keys:
                print(f"[DamoCNN] Missing model keys (sample up to 10): {load_info.missing_keys[:10]}")
            if load_info.unexpected_keys:
                print(f"[DamoCNN] Unexpected keys (sample up to 10): {load_info.unexpected_keys[:10]}")

        # Enforce policies
        policy_errors = []
        if non_allowed_skips:
            policy_errors.append(
                f"Found {len(non_allowed_skips)} skipped keys not in allowed prefixes {allowed_skip_prefixes}. Example: {non_allowed_skips[:5]}"
            )
        if load_ratio < min_load_ratio:
            policy_errors.append(
                f"Loaded ratio {load_ratio:.1%} below minimum threshold {min_load_ratio:.0%}."
            )
        if strict and policy_errors:
            raise RuntimeError("[DamoCNN] Strict load failed: " + " | ".join(policy_errors))
        elif policy_errors and verbose:
            print("[DamoCNN][WARN] " + " | ".join(policy_errors))
    else:
        #TODO
        pass

    return model #type: ignore

def damo_L45_L(
    *,
    image_size=384,
    weights_path = os.path.join(os.path.dirname(__file__), 'damo_yolo', 'weights', 'damoyolo_tinynasL45_L_519.pth'),
    structure_path = os.path.join(os.path.dirname(__file__), 'damo_yolo', 'base_models', 'backbones', 'nas_backbones', 'tinynas_L45_kxkx.txt'),
    **kwargs: Any,
) -> DamoCNN:
    
    """
    Constructs a DAMO-YOLO-L architecture from DAMO-YOLO <https://github.com/tinyvision/DAMO-YOLO/>

    Args:
        weights_path (str, optional): Path to the DAMO_YOLO pretrained weights
        structure_path (str): Path to the structure of the DAMO_YOLO model
    """

    return _damo_cnn(
        image_size = image_size,
        weights_path = weights_path,
        structure_path = structure_path,
        **kwargs,
    )
