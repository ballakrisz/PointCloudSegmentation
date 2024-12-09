from base64 import encode
import torch
import torch.nn as nn
from ..backbone import PointViT, PointNet2Decoder, PointNetFPModule
from ..layers import furthest_point_sample
from ..build import MODELS
import logging

@MODELS.register_module()
class pcsHead(nn.Module):
    def __init__(self, in_channels=3, num_parts=50, drop_prob = 0.2, **kwargs):
        super().__init__()
        """
        Classification head for he Point Cloud Segmentation Network

        Args:
            in_channels (int): Number of input channels
            num_parts (int): Number of parts in the shape net part dataset
            drop_prob (float): Dropout probability
        """

        # MLP layers for local feature extraction
        self.local_mlp = nn.Sequential(
            nn.Conv1d(in_channels, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(drop_prob)
        )

        # MLP layers for global feature extraction
        self.global_mlp = nn.Sequential(
            nn.Conv1d(in_channels, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=1),
            nn.ReLU(),
        )
        
        # Attention-based fusion of global and local features
        self.attention_layer = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)

        # Final classifier (1x1 convolution) to predict part labels
        self.classifier = nn.Conv1d(256, num_parts, kernel_size=1)

        # Residual connections
        self.residual_block = nn.Conv1d(in_channels, 256, kernel_size=1)

    def get_num_layers(self):
        return 7

    def forward(self, point_features):
        """
        Forward pass for the refined classification head.
        
        Args:
            point_features (torch.Tensor): Input features for each point. 
                                           Shape: (B, N, F), where:
                                           - B: Batch size
                                           - N: Number of points
                                           - F: Number of features per point
        
        Returns:
            torch.Tensor: Part label probabilities for each point.
                          Shape: (B, N, num_classes)
        """
        # Transpose to shape (B, F, N) for 1D convolutions
        x = point_features.permute(0, 2, 1)

        # Local feature extraction
        local_features = self.local_mlp(x)

        # Global feature aggregation
        global_features = torch.max(x, dim=2, keepdim=True)[0]  # (B, F, 1)
        global_features = self.global_mlp(global_features)

        # Apply global feature to each point
        global_features = global_features.repeat(1, 1, local_features.size(2))  # (B, F, N)
        fused_features = torch.cat([local_features, global_features], dim=1)  # (B, 2*F, N)
        
        # Apply attention to fused features
        fused_features = fused_features.permute(0, 2, 1)  # (B, N, 2*F)
        attn_output, _ = self.attention_layer(fused_features, fused_features, fused_features)
        attn_output = attn_output.permute(0, 2, 1)  # (B, 2*F, N)

        # Apply residual connection: Add the original features (x) to the attention output
        residual = self.residual_block(x)  
        attn_output = attn_output + residual 

        # Final classifier (convolution for classification)
        output = self.classifier(attn_output)

        # Transpose back to shape (B, N, num_classes) for final output
        output = output.permute(0, 2, 1)

        return output