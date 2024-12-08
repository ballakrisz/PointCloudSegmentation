'''
Copyright 2022@Pixel2Point
File Description: Vision Transformer for Point Cloud Segmentation
'''
from base64 import encode
import torch
import torch.nn as nn
from .head import pcsHead
from ..backbone import PointViT, PointNet2Decoder, PointNetFPModule
from ..backbone.pointnetv2 import PointNet2PartDecoder
from ..layers import furthest_point_sample
from ..build import MODELS
import logging


@MODELS.register_module()
class PointVitSeg(nn.Module):
    def __init__(self,
                 in_channels=6, num_classes=40,
                 encoder_dim=768,  depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 embed_args={'NAME': 'groupembed',
                             'num_groups': 256,
                             'group_size': 32,
                             'embed_dim': 256,
                             'subsample': 'fps',
                             'group': 'knn',
                             'feature_type': 'fj'},
                 conv_args={},
                 norm_args={'norm': 'ln', 'eps': 1.0e-6},
                 act_args={'act': 'gelu'},
                 posembed_norm_args=None,
                 num_points=None, 
                 fp_mlps=None,
                 **kwargs
                 ):
        """ViT for point cloud segmentation

        Args:
            cfg (dict): configuration
        """
        super().__init__()
        if kwargs:
            logging.warning(f"kwargs: {kwargs} are not used in {__class__.__name__}")
        self.encoder = PointViT(
            in_channels,
            encoder_dim, depth,
            num_heads, mlp_ratio, qkv_bias,
            drop_rate, attn_drop_rate, drop_path_rate,
            embed_args, norm_args, act_args, posembed_norm_args
        )
        skip_channel_list = [in_channels]*(len(num_points)+1) + [encoder_dim]
        radius = [4, 8, 16]  # Example radii for multi-scale grouping
        num_samples = [32, 64, 128]  # Example numbers of samples per radius
        group_args = embed_args  # Assuming group_args matches embed_args
        fp_mlps = fp_mlps if fp_mlps else [[256, 128], [128, 64]]  # Example fallback for `fp_mlps`

        # Initialize the first function
        self.decoder = PointNet2PartDecoder(
            in_channels=6,
            radius=radius,
            num_samples=num_samples,
            group_args=group_args,
            conv_args=conv_args,
            norm_args=norm_args,
            act_args=act_args,
            mlps=[[128, 256], [256, 512], [256, 512]],  # Example MLP structure
            blocks=[1, 2, 3],  # Example block structure
            width=encoder_dim,  # Mapping `encoder_dim` to `width`
            strides=[4, 4, 4, 4],
            layers=depth,  # Mapping `depth` to `layers`
            fp_mlps=fp_mlps,
            decoder_layers=1,  # Example
            decocder_aggr_args=None,  # Optional
            width_scaling=2,
            radius_scaling=2,
            nsample_scaling=1,
            use_res=False,
            stem_conv=False,
            double_last_channel=False
        )
        self.head = pcsHead(in_channels=fp_mlps[0][0], num_classes=16, num_parts=50)#TODO SceneSegHeadPointNet(num_classes=num_classes, in_channels=fp_mlps[0][0])
        self.num_points = num_points 
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, xyz, features):
        center_xyz, l_feature = self.encoder(xyz, features)
        
        # to B, C, N
        l_feature = l_feature[:, 1:, :].transpose(1, 2).contiguous()
        
        # generate l_xyz
        l_xyz, l_features = [xyz], [features]
        for npoints in self.num_points[:-1]:
            idx = furthest_point_sample(xyz, npoints).long()
            l_xyz.append(torch.gather(xyz, 1, idx.unsqueeze(-1).expand(-1, -1, 3)))
            l_features.append(torch.gather(features, -1, idx.unsqueeze(1).expand(-1, features.shape[1], -1))
)
        l_xyz.append(center_xyz)
        l_features.append(l_feature)
        
        """Debug
        for i in l_xyz:
            print(i.shape)
            
        for i in l_features:
            print(i.shape)
        """    
        up_features = self.decoder(l_xyz, l_features)
        # up_features = self.decoder(xyz, l_xyz, features, l_features)
        return self.head(up_features)
