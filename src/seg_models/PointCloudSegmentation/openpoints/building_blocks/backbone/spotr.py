from typing import List, Type
import logging
import torch
import torch.nn as nn
from ..build import MODELS
from ..layers import create_convblock1d, create_convblock2d, create_act, CHANNEL_MAP, \
    create_grouper, furthest_point_sample, random_sample, three_interpolation
    
from torch.autograd import Variable
from einops import rearrange, repeat


def get_aggregation_features(p, dp, f, fj, feature_type='dp_fj'):
    """
    Aggregates features based on the specified feature type.

    Args:
        p (torch.Tensor): The original points tensor of shape (B, C, N), 
                          where B is the batch size, C is the number of channels, and N is the number of points.
        dp (torch.Tensor): The differences between neighboring points and the central point, 
                           of shape (B, C, N, K), where K is the number of neighbors.
        f (torch.Tensor): The central point features of shape (B, F, N), 
                          where F is the number of feature channels.
        fj (torch.Tensor): The neighboring points' features of shape (B, F, N, K).
        feature_type (str): Specifies the type of aggregation to perform. Options include:
                            - 'dp_fj': Concatenate `dp` and `fj`.
                            - 'dp_fj_df': Concatenate `dp`, `fj`, and the feature difference `df`.
                            - 'pi_dp_fj_df': Concatenate original points `p`, `dp`, `fj`, and `df`.
                            - 'dp_df': Concatenate `dp` and `df`.

    Returns:
        torch.Tensor: The aggregated features based on the selected feature type.
    """
    if feature_type == 'dp_fj':
        # Concatenate the differences in position (`dp`) with neighboring features (`fj`).
        fj = torch.cat([dp, fj], dim=1)
    
    elif feature_type == 'dp_fj_df':
        # Calculate the feature difference `df` between neighboring features (`fj`) and central features (`f`).
        df = fj - f.unsqueeze(-1)  # Expand `f` to match the shape of `fj`.
        # Concatenate `dp`, `fj`, and `df`.
        fj = torch.cat([dp, fj, df], dim=1)
    
    elif feature_type == 'pi_dp_fj_df':
        # Calculate the feature difference `df`.
        df = fj - f.unsqueeze(-1)
        # Expand and repeat the original points (`p`) to match the neighborhood dimension.
        p_expanded = p.transpose(1, 2).unsqueeze(-1).expand(-1, -1, -1, df.shape[-1])
        # Concatenate `p`, `dp`, `fj`, and `df`.
        fj = torch.cat([p_expanded, dp, fj, df], dim=1)
    
    elif feature_type == 'dp_df':
        # Calculate the feature difference `df`.
        df = fj - f.unsqueeze(-1)
        # Concatenate `dp` and `df`.
        fj = torch.cat([dp, df], dim=1)

    return fj



def new_attention(x_i, y_j=None, attn=None, aux_attn=None, tau=1):
    """
    Computes a new attention mechanism for input features, either in a batch-wise context 
    or with additional auxiliary attention.

    Args:
        x_i (torch.Tensor): Input tensor, shape can be (B, D, N) or (B, D, N, M) depending on context.
                            - B: Batch size
                            - D: Feature/channel dimension
                            - N: Number of source elements
                            - M: Number of target elements (optional in 4D case)
        y_j (torch.Tensor, optional): Target tensor for attention computation, shape (B, M, D).
        attn (torch.Tensor, optional): Precomputed attention weights, shape (B, D, N, M) in the 4D case.
        aux_attn (torch.Tensor, optional): Auxiliary attention weights, shape (B, N, M).
        tau (float, optional): Temperature scaling factor for attention normalization, default is 1.

    Returns:
        torch.Tensor: Attention-modulated output tensor, shape depends on input:
                      - (B, D, M) for 3D input.
                      - (B, D, N) for 4D input after reduction along the last axis.
    """
    if len(x_i.shape) == 3:  # Case for 3D input tensors (batch, channels, source elements)
        # Compute attention matrix between `y_j` and `x_i`.
        attn = torch.bmm(y_j.transpose(1, 2).contiguous(), x_i).detach()  # Shape: (B, M, N)
        attn = nn.functional.softmax(attn, -1)  # Apply softmax along the last dimension (N).
        attn = attn * aux_attn  # Element-wise modulation by auxiliary attention if provided.

        # Weighted sum of `x_i` based on attention.
        out2 = torch.bmm(x_i, attn.transpose(1, 2).contiguous())  # Shape: (B, D, M)
        return out2  # Return the attention-modulated features.

    else:  # Case for 4D input tensors (batch, channels, source elements, target elements)
        b, d, n_s, n_g = x_i.shape

        # Normalize the precomputed attention weights with softmax along the last dimension (target elements).
        channel_attn = nn.functional.softmax(attn / tau, -1)  # Shape: (B, D, N_s, N_g)
        out1 = channel_attn * x_i  # Apply attention weights to input features (element-wise).

        if aux_attn is not None:
            # Modulate by auxiliary attention if provided. Expand aux_attn to match channel dimensions.
            out1 = out1 * aux_attn.unsqueeze(1)  # Shape: (B, D, N_s, N_g)

        # Reduce the output along the last axis (target elements) by summing over it.
        return out1.sum(-1)  # Shape: (B, D, N_s)


class FeaturePropogation(nn.Module):
    """
    Feature Propagation module in PointNet++ for upsampling and aggregating features.
    """

    def __init__(self, mlp, upsample=True, norm_args={'norm': 'bn1d'}, act_args={'act': 'relu'}):
        """
        Args:
            mlp: List specifying channel dimensions for MLP layers [input_channels, intermediate_channels, output_channels].
            upsample: Boolean indicating whether to perform feature upsampling.
            norm_args: Dictionary specifying normalization parameters.
            act_args: Dictionary specifying activation function parameters.
        """
        super().__init__()
        if not upsample:
            # For feature merging (global features)
            self.linear2 = nn.Sequential(nn.Linear(mlp[0], mlp[1]), nn.ReLU(inplace=True))
            mlp[1] *= 2  # Adjusting channel size for concatenation
            linear1 = [
                create_convblock1d(mlp[i], mlp[i + 1], norm_args=norm_args, act_args=act_args)
                for i in range(1, len(mlp) - 1)
            ]
            self.linear1 = nn.Sequential(*linear1)
        else:
            # For feature upsampling (local features)
            convs = [
                create_convblock1d(mlp[i], mlp[i + 1], norm_args=norm_args, act_args=act_args)
                for i in range(len(mlp) - 1)
            ]
            self.convs = nn.Sequential(*convs)

        # Pooling function for computing global features
        self.pool = lambda x: torch.mean(x, dim=-1, keepdim=False)

    def forward(self, pf1, pf2=None):
        """
        Forward pass for propagating features.
        Args:
            pf1: Tuple (points, features) for the first input.
            pf2: Tuple (points, features) for the second input (used for upsampling).
        """
        if pf2 is None:
            # Global feature merging
            _, f = pf1  # Extract features
            f_global = self.pool(f)  # Compute global features
            # Concatenate global features and apply linear layers
            f = torch.cat(
                (f, self.linear2(f_global).unsqueeze(-1).expand(-1, -1, f.shape[-1])), dim=1
            )
            f = self.linear1(f)
        else:
            # Feature upsampling
            p1, f1 = pf1  # Source points and features
            p2, f2 = pf2  # Target points and features
            # Interpolate features from p2 to p1
            interpolated = three_interpolation(p1, p2, f2)
            # Combine interpolated and existing features
            f = self.convs(torch.cat((f1, interpolated), dim=1) if f1 is not None else interpolated)
        return f


class LPAMLP(nn.Module):
    """
    LPA + MLP block for local feature aggregation with attention mechanisms.
    """

    def __init__(self, in_channels, norm_args=None, act_args=None,
                 aggr_args={'feature_type': 'dp_fj', "reduction": 'max'},
                 group_args={'NAME': 'ballquery'}, conv_args=None,
                 expansion=1, use_res=True, num_posconvs=2, less_act=False,
                 gamma=16, num_gp=16, tau_delta=1, **kwargs):
        """
        Args:
            in_channels: Number of input feature channels.
            norm_args: Dictionary specifying normalization parameters.
            act_args: Dictionary specifying activation function parameters.
            aggr_args: Arguments for feature aggregation (e.g., feature type and reduction method).
            group_args: Arguments for grouping points (e.g., ball query settings).
            conv_args: Arguments for convolution operations.
            expansion: Expansion factor for the FFN layer.
            use_res: Whether to use residual connections.
            num_posconvs: Number of point-wise convolutions in the FFN.
            less_act: Whether to use fewer activations in the FFN.
            gamma, num_gp, tau_delta: Additional parameters for controlling attention mechanisms.
        """
        super().__init__()

        self.gamma = gamma
        self.num_gp = num_gp
        self.tau_delta = tau_delta
        self.use_res = use_res
        self.feature_type = aggr_args['feature_type']  # Determines how features are aggregated

        # Determine input channel size after feature mapping
        channels = [CHANNEL_MAP[self.feature_type](in_channels)] + [in_channels] * 2

        # Attention block for local feature enhancement
        self.attn_local = create_convblock2d(channels[0], channels[-1], norm_args=norm_args, **conv_args)

        # Convolution layers for processing local features
        self.convs = nn.Sequential(
            *[create_convblock2d(channels[i], channels[i + 1], norm_args=norm_args,
                                 act_args=None if i == len(channels) - 2 else act_args, **conv_args)
              for i in range(len(channels) - 1)]
        )

        # Point grouping layer (e.g., ball query)
        self.grouper = create_grouper(group_args)

        # Feed-forward network (FFN) for feature transformation
        ffn_channels = [in_channels, int(in_channels * expansion), in_channels] if num_posconvs > 1 else [in_channels] * num_posconvs
        self.ffn = nn.Sequential(
            *[create_convblock1d(ffn_channels[i], ffn_channels[i + 1], norm_args=norm_args,
                                 act_args=act_args if i != len(ffn_channels) - 2 else None, **conv_args)
              for i in range(len(ffn_channels) - 1)]
        )

        # Activation function for non-linear transformations
        self.act = create_act(act_args)

        # Learnable parameter for controlling attention scaling
        self.alpha = nn.Parameter(torch.zeros((1,), dtype=torch.float32))

    def forward(self, pf):
        """
        Forward pass for LPA + MLP.
        Args:
            pf: Tuple of (points, features).
        """
        p, f = pf  # Points and features
        identity = f  # Save original features for residual connection

        # Perform grouping and feature aggregation
        dp, fj = self.grouper(p, p, f)
        fj = get_aggregation_features(p, dp, f, fj, self.feature_type)

        # Apply attention to the aggregated features
        f = new_attention(self.convs(fj), attn=self.attn_local(fj))
        f = self.act(f + identity)  # Add residual connection

        # Feed-forward network for feature transformation
        identity = f  # Save intermediate features for residual connection
        f = self.ffn(f)
        if f.shape[-1] == identity.shape[-1] and self.use_res:
            f += identity  # Residual connection
        f = self.act(f)
        return p, f

@MODELS.register_module()
class SPoTrPartDecoder(nn.Module):
    """
    A decoder module for part segmentation, using hierarchical feature propagation.
    """

    def __init__(self,
                 encoder_channel_list: List[int],
                 decoder_layers: int = 5,
                 decoder_blocks: List[int] = [1, 1, 1, 1],
                 decoder_strides: List[int] = [4, 4, 4, 4],
                 act_args: str = 'relu',
                 num_classes: int = 50,
                 **kwargs):
        """
        Args:
            encoder_channel_list: List of channel sizes from the encoder.
            decoder_layers: Number of layers in each decoder block.
            decoder_blocks: Number of blocks in each decoder stage.
            decoder_strides: List of strides for each decoder block.
            act_args: Activation function type (e.g., 'relu').
            num_classes: Number of output classes (e.g., part categories).
            **kwargs: Additional arguments such as normalization, convolution configurations.
        """
        super().__init__()

        # Initialize decoder-specific parameters
        self.decoder_layers = decoder_layers
        self.in_channels = encoder_channel_list[-1]  # Start with the last encoder channel size
        skip_channels = encoder_channel_list[:-1]  # Skip channels from the encoder (except last)
        fp_channels = encoder_channel_list[:-1]  # Feature propagation channels

        # Decoder block configuration
        self.conv_args = kwargs.get('conv_args', {'kernel_size': 1})  # Convolution argument (default kernel size = 1)
        radius_scaling = kwargs.get('radius_scaling', 2)  # Scaling factor for radius
        nsample_scaling = kwargs.get('nsample_scaling', 1)  # Scaling factor for sampling
        block = kwargs.get('block', 'LPAMLP')  # Type of block used for the decoder (default: 'LPAMLP')

        # Resolve block type
        if isinstance(block, str):
            block = eval(block)  # Evaluate block name if it's a string (e.g., 'LPAMLP')

        # Decoder structure
        self.blocks = decoder_blocks
        self.strides = decoder_strides
        self.norm_args = kwargs.get('norm_args', {'norm': 'bn'})  # Normalization type (default: batch norm)
        self.act_args = kwargs.get('act_args', {'act': 'relu'})  # Activation function type (default: ReLU)
        self.expansion = kwargs.get('expansion', 4)  # Expansion factor for the decoder layers
        radius = kwargs.get('radius', 0.1)  # Default radius
        nsample = kwargs.get('nsample', 16)  # Default number of samples
        self.radii = self._to_full_list(radius, radius_scaling)  # Generate full list of radii
        self.nsample = self._to_full_list(nsample, nsample_scaling)  # Generate full list of sample numbers
        self.num_classes = num_classes  # Number of output classes (part categories)
        self.use_res = kwargs.get('use_res', True)  # Whether to use residual connections
        group_args = kwargs.get('group_args', {'NAME': 'ballquery'})  # Grouping arguments for feature aggregation
        self.aggr_args = kwargs.get('aggr_args', {'feature_type': 'dp_fj', "reduction": 'max'})  # Aggregation settings

        # Define global convolution layers (to extract high-level features)
        self.global_conv2 = nn.Sequential(
            create_convblock1d(encoder_channel_list[-1], 128,
                                norm_args=None,
                                act_args=act_args)
        )
        self.global_conv1 = nn.Sequential(
            create_convblock1d(encoder_channel_list[-2], 64,
                                norm_args=None,
                                act_args=act_args)
        )
        skip_channels[0] += 64 + 128 + 50  # Adding space for shape category labels

        # Build decoder blocks for each stage
        n_decoder_stages = len(fp_channels)
        decoder = [[] for _ in range(n_decoder_stages)]  # List to hold blocks for each stage
        for i in range(-1, -n_decoder_stages - 1, -1):
            group_args.radius = self.radii[i]  # Set the radius for this block
            group_args.nsample = self.nsample[i]  # Set the number of samples for this block
            decoder[i] = self._make_dec(
                skip_channels[i], fp_channels[i], group_args=group_args, block=block, blocks=self.blocks[i])

        # Combine the blocks into a sequential model
        self.decoder = nn.Sequential(*decoder)
        self.out_channels = fp_channels[-n_decoder_stages]  # Set the output channel size

    def _make_dec(self, skip_channels, fp_channels, group_args=None, block=None, blocks=1):
        """
        Create the decoder for a given stage, using feature propagation and blocks.
        
        Args:
            skip_channels: Channels from the encoder to be concatenated at each stage.
            fp_channels: Feature propagation channels at this stage.
            group_args: Grouping arguments for point grouping and feature aggregation.
            block: Type of block to use (e.g., 'LPAMLP').
            blocks: Number of blocks at this stage.
        """
        layers = []
        radii = group_args.radius  # Radius for this stage
        nsample = group_args.nsample  # Number of samples for this stage

        # Define multi-layer perceptron (MLP) for feature propagation
        mlp = [skip_channels + self.in_channels] + [fp_channels] * self.decoder_layers
        layers.append(FeaturePropogation(mlp, act_args=self.act_args))  # Feature propagation layer
        self.in_channels = fp_channels  # Update input channels for next block

        # Add additional blocks (if any) to the decoder
        for i in range(1, blocks):
            group_args.radius = radii[i]  # Set the radius for this block
            group_args.nsample = nsample[i]  # Set the number of samples for this block
            layers.append(block(self.in_channels,
                                aggr_args=self.aggr_args,
                                norm_args=self.norm_args, act_args=self.act_args, group_args=group_args,
                                conv_args=self.conv_args, expansion=self.expansion,
                                use_res=self.use_res))  # Add the block to the decoder layers

        return nn.Sequential(*layers)  # Return the sequential decoder stage

    def _to_full_list(self, param, param_scaling=1):
        """
        Ensure that the provided parameter (radius or nsample) is converted to a full list.
        If it's a scalar, it will be expanded according to the number of decoder blocks.

        Args:
            param: The parameter (either radius or nsample).
            param_scaling: Scaling factor for the parameter.
        """
        param_list = []
        if isinstance(param, List):
            # If param is already a list, ensure it's the correct length for each block
            for i, value in enumerate(param):
                value = [value] if not isinstance(value, List) else value
                if len(value) != self.blocks[i]:
                    value += [value[-1]] * (self.blocks[i] - len(value))
                param_list.append(value)
        else:
            # If param is scalar, create a list with scaling factors for each block
            for i, stride in enumerate(self.strides):
                if stride == 1:
                    param_list.append([param] * self.blocks[i])
                else:
                    param_list.append([param] + [param * param_scaling] * (self.blocks[i] - 1))
                    param *= param_scaling
        return param_list

    def get_num_layers(self):
        """
        Calculate the total number of layers in the decoder, including global and decoder layers.
        """
        # Global convolution layers
        global_layers = len(self.global_conv1) + len(self.global_conv2)

        # Decoder layers: Each decoder block contains several layers (defined by self.decoder_layers)
        decoder_layers = 0
        for i in range(len(self.blocks)):
            decoder_layers += self.decoder_layers * self.blocks[i]

        # Total layers = global layers + decoder layers
        total_layers = global_layers + decoder_layers
        return total_layers

    def forward(self, p, f, cls_label):
        """
        Forward pass through the SPoTrPartDecoder.
        
        Args:
            p: List of points at different stages (hierarchical resolution).
            f: List of features at different stages.
            cls_label: Class label (used for shape classification).
        """
        B, N = p[0].shape[0:2]

        # Apply global convolutions to extract global features
        emb1 = self.global_conv1(f[-2])
        emb1 = emb1.max(dim=-1, keepdim=True)[0]  # Aggregate over points (max pooling)
        emb2 = self.global_conv2(f[-1])
        emb2 = emb2.max(dim=-1, keepdim=True)[0]  # Aggregate over points (max pooling)

        # Create one-hot encoded class labels and concatenate with global features
        cls_one_hot = torch.zeros((B, self.num_classes), device=p[0].device)
        cls_one_hot = cls_one_hot.scatter_(1, cls_label, 1).unsqueeze(-1)
        cls_one_hot = torch.cat((emb1, emb2, cls_one_hot), dim=1)
        cls_one_hot = cls_one_hot.expand(-1, -1, N)  # Expand across all points

        # Perform feature propagation and decoding for each stage
        for i in range(-1, -len(self.decoder), -1):
            f[i - 1] = self.decoder[i][1:](
                [p[i-1], self.decoder[i][0]([p[i - 1], f[i - 1]], [p[i], f[i]])])[1]

        # Final decoding using class label and concatenated features
        f[-len(self.decoder) - 1] = self.decoder[0][1:](
            [p[0], self.decoder[0][0]([p[0], torch.cat([cls_one_hot, f[0]], 1)], [p[1], f[1]])])[1]

        return f[-len(self.decoder) - 1]  # Return the final decoded feature
