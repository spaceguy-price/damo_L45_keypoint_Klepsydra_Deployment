# model.py
# This file has been simplified from a larger evaluation repository.
# It is contains the model definition of DAMO L45 L with a Keypoint Head
# It is intended to support the Klepsydra deployment process

# Author: Andrew Price
# andrew.price@epfl.ch
# 19.03.2026

import torch
import torch.nn as nn
from torchvision.ops.misc import MLP
from damo_cnn import damo_L45_L

class MLPWithProjection(nn.Module):
    """
    MLP with a linear projection layer at the end
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: list[int],
        activation_function: nn.Module,
        out_channels: int,
    ):
        """
        Initialize the MLP with projection layer

        Args:
            in_channels (int): Number of input channels
            hidden_channels (list): List of hidden layer dimensions
            activation_function (nn.Module): Activation function to use
            out_channels (int): Number of output channels
        """
        super().__init__()
        self.mlp = MLP(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            activation_layer=activation_function,
        )
        self.projection = nn.Linear(hidden_channels[-1], out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        return self.projection(x)

# --------------------------------------
# CNN FPN to FC Adapter for the Damo CNN
# --------------------------------------
class FPNtoFCAdapter(nn.Module):
    """Concatenates FPN feature maps into a single vector for FC head"""
    def __init__(
            self, 
            in_channels=[128,256,512],
            spatial_sizes=[28,14,7],
            output_dim=728,
            ):
        
        super().__init__()

        self.flatten=nn.Flatten() #(B,C,H,W)->(B, C*H*W)
        
        #Compute total input size after flattening
        total_input_dim = sum([c * (s ** 2) for c, s in zip(in_channels, spatial_sizes)])
        #Linear projection
        self.fc = nn.Linear(total_input_dim, output_dim)

    def forward(self, fpn_outputs):
        """
        Arguments:
            fpn_outputs: Tuple of FPN features (x5,x7,x6)
        Returns:
            Flattended feasure vector (B, output_dim)
        """
        flattened_features = [self.flatten(output) for output in fpn_outputs] #(B, C*H*W)
        concatenated_features = torch.cat(flattened_features, dim=1) #(B, total_input_dim)
        projected_feature = self.fc(concatenated_features) # (B, output_dim)

        return projected_feature

# -----------------------------------
# DAMO YOLO CNN Pose Estimation Model
# -----------------------------------
class DamoPose(nn.Module):
    """
    Pose estimation model using the Damo Yolo architecure
    """

    def __init__(
        self,
        num_hidden_layers: int = 3,
        hidden_layer_dim: int = 800,
        num_keypoints: int = 8,
        # DAMO loader/diagnostics flags (use Python args, not env)
        damo_load_verbose: bool = True,
        damo_load_strict: bool = False,
        damo_min_load_ratio: float = 0.6,
        damo_allowed_skip_prefixes: str = "head.,backbone.block_list.",
        damo_shape_verbose: bool = False,
    ):
        """
        Model initialization

        Args:
            num_hidden_layers (int): Number of hidden layers in the MLPs. Defaults to 3.
            hidden_layer_dim (int): Dimension of hidden layers in the MLPs. Defaults to 800.
            num_keypoints (int, optional): Number of keypoints to regress for PnP. Defaults to 8.
            damo_load_verbose (bool): Whether to print verbose loading info for the Damo model. Defaults to True.
            damo_load_strict (bool): Whether to use strict loading for the Damo model
            damo_min_load_ratio (float): Minimum load ratio for the Damo model weights. Defaults to 0.6.
            damo_allowed_skip_prefixes (str): Comma-separated prefixes to allow when skipping weights during Damo model loading. Defaults to "head.,backbone.block_list.".
            damo_shape_verbose (bool): Whether to print the inferred FPN shapes after loading the Damo model.
        """
        super().__init__()

        # Get the model factory
        #cnn_func, fpn_dim, fpn_spatial, img_size = CNN_MODELS[cnn_model]
        #fpn_dim, fpn_spatial = [128,256,512], [48,24,12]

        # Load the model with explicit DAMO loader flags
        self.model = damo_L45_L(
            damo_load_verbose=damo_load_verbose,
            damo_load_strict=damo_load_strict,
            damo_min_load_ratio=damo_min_load_ratio,
            damo_allowed_skip_prefixes=damo_allowed_skip_prefixes,
        )

        # Dynamically infer FPN channels and spatial sizes from the backbone outputs
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.model.image_size, self.model.image_size)
            feats = self.model(dummy)
            fpn_dim = [f.shape[1] for f in feats]
            fpn_spatial = [f.shape[-1] for f in feats]
        # Optional: print the inferred shapes for visibility (Python flag, not env)
        if damo_shape_verbose:
            print(f"[DamoPose] Inferred FPN channels: {fpn_dim}, spatial sizes: {fpn_spatial}")

        # Remove classification head
        # Note that the head has been commented out of the forward call in the cnn_models.py file
        self.model.head = nn.Identity() #type: ignore

        #BUILD FPN->HEAD adapter
        # If different image sizes are required, the FPN spatial sizes must be adjusted
        self.adapter = FPNtoFCAdapter(
            in_channels=fpn_dim,
            spatial_sizes=fpn_spatial,  # derived from the actual model outputs
            output_dim=hidden_layer_dim,  # project into the head hidden dimension size
        )

        # --Implement the head--
        self.output = MLPWithProjection(
            in_channels=hidden_layer_dim, 
            hidden_channels=[hidden_layer_dim] * num_hidden_layers, 
            activation_function=nn.ReLU, #type: ignore
            out_channels=num_keypoints*2
        )
    

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model

        Args:
            x (torch.Tensor): Input image tensor

        Returns:
            torch.Tensor: Output tensor
        """
        fpn_outputs = self.model(x)
        x = self.adapter(fpn_outputs)
        x = self.output(x)
        return x.reshape(x.size(0), int(x.size(1)/2), 2) #[B, N*2] -> [B,N,2]

    def load_from_pretrained(self, path: str) -> None:
        """
        Load model weights from a pretrained model

        Args:
            path (str): Path to the pretrained model weights

        Returns:
            None
        """
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        self.load_state_dict(state_dict)

    def save(self, path: str) -> None:
        """
        Save the model weights

        Args:
            path (str): Path to save the model weights

        Returns:
            None
        """
        torch.save(self.state_dict(), path)
