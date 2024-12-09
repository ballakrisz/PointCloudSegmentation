""" Vision Transformer (ViT) for Point Cloud Understanding in PyTorch
Hacked together by / Copyright 2020, Ross Wightman
Modified to 3D application by / Copyright 2022@Pix4Point team
"""
import logging
from typing import List
import torch
import torch.nn as nn
from ..layers import create_norm, create_linearblock, create_convblock1d, three_interpolation, \
    furthest_point_sample, random_sample
from ..layers.attention import Block
from .pointnext import FeaturePropogation
from ..build import MODELS, build_model_from_cfg


@MODELS.register_module()
class PointViT(nn.Module):
    """ Point Vision Transformer ++: with early convolutions
    A transformer-based model for point cloud processing with added convolutions before the transformer layers.
    """
    def __init__(self,
                 in_channels=3,
                 embed_dim=384, depth=12,
                 num_heads=6, mlp_ratio=4., qkv_bias=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 embed_args={'NAME': 'PointPatchEmbed', 
                             'num_groups': 256,
                             'group_size': 32,
                             'subsample': 'fps', 
                             'group': 'knn', 
                             'feature_type': 'fj',
                             'norm_args': {'norm': 'in2d'},
                             }, 
                 norm_args={'norm': 'ln', 'eps': 1.0e-6},
                 act_args={'act': 'gelu'},
                 add_pos_each_block=True,
                 global_feat='cls,max',
                 distill=False, 
                 **kwargs
                 ):
        """
        Args:
            in_channels (int): number of input channels. Default: 6. (p + rgb)
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_args (dict): Arguments for embedding layer configuration
            norm_args (dict): Arguments for normalization layers
            act_args (dict): Arguments for activation functions
            add_pos_each_block (bool): Whether to add positional encoding at each transformer block
            global_feat (str): Type of global feature aggregation ('cls', 'max', etc.)
            distill (bool): Whether to use distillation tokens
        """
        super().__init__()

        # Initialize the number of features and embedding dimension
        self.num_features = self.embed_dim = embed_dim
        
        # Setup the embedding layer based on provided arguments
        embed_args.in_channels = in_channels
        embed_args.embed_dim = embed_dim

        # Create the patch embedding layer (P3Embed)
        self.patch_embed = build_model_from_cfg(embed_args)

        # Initialize the class token and positional encoding tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.embed_dim))

        # Positional encoding layer
        self.pos_embed = nn.Sequential(
            create_linearblock(3, 128, norm_args=None, act_args=act_args),
            nn.Linear(128, self.embed_dim)
        )

        # Projection layer to match patch embedding output dimensions if necessary
        if self.patch_embed.out_channels != self.embed_dim: 
            self.proj = nn.Linear(self.patch_embed.out_channels, self.embed_dim)
        else:
            self.proj = nn.Identity()

        # Whether to add positional encoding at each block
        self.add_pos_each_block = add_pos_each_block
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth decay rule (for drop path)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  
        self.depth = depth
        
        # Transformer blocks (list of layers)
        self.blocks = nn.ModuleList([
            Block(
                dim=self.embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_args=norm_args, act_args=act_args
            )
            for i in range(depth)])

        # Normalization layer
        self.norm = create_norm(norm_args, self.embed_dim)
        
        # Parse global feature types
        self.global_feat = global_feat.split(',')
        self.out_channels = len(self.global_feat) * embed_dim
        self.distill_channels = embed_dim

        # Modify the channel list to match embedding dimension
        self.channel_list = self.patch_embed.channel_list
        self.channel_list[-1] = embed_dim

        # Setup distillation tokens if distillation is enabled
        if distill:
            self.dist_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
            self.dist_pos = nn.Parameter(torch.randn(1, 1, self.embed_dim))
            self.n_tokens = 2
        else:
            self.dist_token = None
            self.n_tokens = 1

        # Initialize model weights
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize specific parameters using a normal distribution
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.cls_pos, std=.02)
        if self.dist_token is not None:
            torch.nn.init.normal_(self.dist_token, std=.02)
            torch.nn.init.normal_(self.dist_pos, std=.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        # Weight initialization for different layers
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        # Specify parameters that should not have weight decay (e.g., tokens)
        return {'cls_token', 'dist_token', 'dist_token'}

    def get_num_layers(self):
        # Return the number of transformer blocks (depth)
        return self.depth

    def forward(self, p, x=None):
        """ Forward pass for PointViT
        Args:
            p (Tensor): Point cloud input (coordinates)
            x (Tensor, optional): Features associated with the points (e.g., colors)

        Returns:
            p_list, x_list, x: Processed point cloud, features, and output
        """
        # Check if p is a dictionary and split it into pos and x if needed
        if hasattr(p, 'keys'): 
            p, x = p['pos'], p['x'] if 'x' in p.keys() else None

        # If no features provided, use the positions as features
        if x is None:
            x = p.clone().transpose(1, 2).contiguous()

        # Apply patch embedding to the point cloud and features
        p_list, x_list = self.patch_embed(p, x)
        
        # Extract the last set of point positions and features for further processing
        center_p, x = p_list[-1], self.proj(x_list[-1].transpose(1, 2))

        # Compute positional embeddings
        pos_embed = self.pos_embed(center_p)

        # Add class token and position token
        pos_embed = [self.cls_pos.expand(x.shape[0], -1, -1), pos_embed]
        tokens = [self.cls_token.expand(x.shape[0], -1, -1), x]
        if self.dist_token is not None:
            pos_embed.insert(1, self.dist_pos.expand(x.shape[0], -1, -1)) 
            tokens.insert(1, self.dist_token.expand(x.shape[0], -1, -1)) 
        pos_embed = torch.cat(pos_embed, dim=1)
        x = torch.cat(tokens, dim=1)

        # Add positional embeddings to the input and pass through transformer blocks
        if self.add_pos_each_block:
            for block in self.blocks:
                x = block(x + pos_embed)
        else:
            x = self.pos_drop(x + pos_embed)
            for block in self.blocks:
                x = block(x)

        # Normalize the output
        x = self.norm(x)
        return p_list, x_list, x

    def forward_cls_feat(self, p, x=None):
        """ Forward pass to extract class features
        Args:
            p (Tensor): Point cloud input
            x (Tensor, optional): Features associated with points

        Returns:
            global_features: Extracted global features from cls and max operations
        """
        # Get features from forward pass
        _, _, x = self.forward(p, x)

        # Extract token features (exclude dist_token if available)
        token_features = x[:, self.n_tokens:, :]

        # Aggregate global features (cls, max, mean)
        cls_feats = []
        for token_type in self.global_feat:
            if 'cls' in token_type:
                cls_feats.append(x[:, 0, :])  # Use class token
            elif 'max' in token_type:
                cls_feats.append(torch.max(token_features, dim=1, keepdim=False)[0])  # Max pooling
            elif token_type in ['avg', 'mean']:
                cls_feats.append(torch.mean(token_features, dim=1, keepdim=False))  # Mean pooling

        # Concatenate global features
        global_features = torch.cat(cls_feats, dim=1)
        
        # If distillation is used, return distillation token features as well
        if self.dist_token is not None and self.training:
            return global_features, x[:, 1, :]
        else: 
            return global_features

    def forward_seg_feat(self, p, x=None):
        """ Forward pass to extract segmentation features
        Args:
            p (Tensor): Point cloud input
            x (Tensor, optional): Features associated with points

        Returns:
            p_list, x_list: Processed point cloud and features
        """
        p_list, x_list, x = self.forward(p, x)
        # Return the features with the last feature list transposed
        x_list[-1] = x.transpose(1, 2)
        return p_list, x_list
