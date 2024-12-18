import torch
import torch.nn as nn
from .head import pcsHead
from ..backbone.spotr import SPoTrPartDecoder
from ..backbone import PointViT, PointNet2Decoder, PointNetFPModule
from ..backbone.pointnetv2 import PointNet2PartDecoder
from ..layers import furthest_point_sample
from ..build import MODELS
import logging

@MODELS.register_module()
class PCS(nn.Module):
    def __init__(self,
                 in_channels=6, num_classes=50,
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
                 blocks= [1, 1, 1, 1],
                 strides= [4, 4, 4, 4],
                 **kwargs
                 ):
        """
        ViT for point cloud segmentation, these parameters are specidied in the pcs.yaml file

        Args:
            in_channels (int): the number of input channels
            num_classes (int): the number of classes
            encoder_dim (int): the number of channels in the encoder
            depth (int): the number of transformer blocks
        """
        super().__init__()
        if kwargs:
            logging.warning(f"kwargs: {kwargs} are not used in {__class__.__name__}")
        
        # Use the PointViT as the encoder
        self.encoder = PointViT(
            in_channels,
            encoder_dim, depth,
            num_heads, mlp_ratio, qkv_bias,
            drop_rate, attn_drop_rate, drop_path_rate,
            embed_args, norm_args, act_args, posembed_norm_args
        )
        # Use the SPoTrPartDecoder as the decoder
        self.decoder = SPoTrPartDecoder(
            self.encoder.channel_list,
            act_args=act_args,
            decoder_blocks=blocks,
            decoder_strides=strides,
            num_classes=num_classes,
            **kwargs
        )
        # Use the pcsHead as the head
        self.head = pcsHead(
            in_channels=2*in_channels,
            num_parts=num_classes, 
            drop_prob=drop_rate
            )
        
    def get_num_layers(self):
        """
        Function to count the number of layers in the model
        """
        return self.encoder.get_num_layers() + self.decoder.get_num_layers() + self.head.get_num_layers()

    def forward(self, xyz, features, class_labels):
            # Extract feature and coordinate encodings at different scales
            p_list, f_list, f = self.encoder(xyz, features)

            # decode the features and coordinates at different scales and embed them
            decoded_feaures = self.decoder(p_list, f_list, class_labels)

            # Concatenate the decoded features with the input coordinates
            xyz_features = torch.cat([xyz, decoded_feaures.permute(0, 2, 1)], dim=2) # [B, N, C]

            # Classify each point (part-wise) with the classification head
            logits = self.head(xyz_features)

            return logits
        
