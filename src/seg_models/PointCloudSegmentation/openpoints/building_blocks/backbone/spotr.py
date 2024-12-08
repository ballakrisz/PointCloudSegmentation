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
    if feature_type == 'dp_fj':
        fj = torch.cat([dp, fj], 1)
    elif feature_type == 'dp_fj_df':
        df = fj - f.unsqueeze(-1)
        fj = torch.cat([dp, fj, df], 1)
    elif feature_type == 'pi_dp_fj_df':
        df = fj - f.unsqueeze(-1)
        fj = torch.cat([p.transpose(1, 2).unsqueeze(-1).expand(-1, -1, -1, df.shape[-1]), dp, fj, df], 1)
    elif feature_type == 'dp_df':
        df = fj - f.unsqueeze(-1)
        fj = torch.cat([dp, df], 1)
    return fj


def new_attention(x_i, y_j=None, attn=None, aux_attn= None, tau= 1):
    if len(x_i.shape) == 3:
        attn = torch.bmm(y_j.transpose(1,2).contiguous(), x_i).detach()
        attn = nn.functional.softmax(attn, -1)#(b,m,n)
        attn = attn*aux_attn
            
        out2 = torch.bmm(x_i, attn.transpose(1,2).contiguous()) #(b,d,m)
        return out2
    else:
        b, d, n_s, n_g = x_i.shape
        channel_attn = (nn.functional.softmax(attn/tau, -1)) #(b, d, n_s n_g)
        channel_attn = channel_attn
        out1 = ((channel_attn))* x_i #(b, d, n_s, n_g) 
        if aux_attn is not None:
            out1 = out1 * aux_attn.unsqueeze(1)#(b,d,n,m) (b 1 n m) -> (b d n m)
        return out1.sum(-1)

class FeaturePropogation(nn.Module):
    """The Feature Propogation module in PointNet++
    """

    def __init__(self, mlp,
                 upsample=True,
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'}
                 ):
        """
        Args:
            mlp: [current_channels, next_channels, next_channels]
            out_channels:
            norm_args:
            act_args:
        """
        super().__init__()
        if not upsample:
            self.linear2 = nn.Sequential(
                nn.Linear(mlp[0], mlp[1]), nn.ReLU(inplace=True))
            mlp[1] *= 2
            linear1 = []
            for i in range(1, len(mlp) - 1):
                linear1.append(create_convblock1d(mlp[i], mlp[i + 1],
                                                  norm_args=norm_args, act_args=act_args
                                                  ))
            self.linear1 = nn.Sequential(*linear1)
        else:
            convs = []
            for i in range(len(mlp) - 1):
                convs.append(create_convblock1d(mlp[i], mlp[i + 1],
                                                norm_args=norm_args, act_args=act_args
                                                ))
            self.convs = nn.Sequential(*convs)

        self.pool = lambda x: torch.mean(x, dim=-1, keepdim=False)

    def forward(self, pf1, pf2=None):
        # pfb1 is with the same size of upsampled points
        if pf2 is None:
            _, f = pf1  # (B, N, 3), (B, C, N)
            f_global = self.pool(f)
            f = torch.cat(
                (f, self.linear2(f_global).unsqueeze(-1).expand(-1, -1, f.shape[-1])), dim=1)
            f = self.linear1(f)
        else:
            p1, f1 = pf1
            p2, f2 = pf2
            if f1 is not None:
                f = self.convs(
                    torch.cat((f1, three_interpolation(p1, p2, f2)), dim=1))
            else:
                f = self.convs(three_interpolation(p1, p2, f2))
        return f
    
# LPA + MLP block
class LPAMLP(nn.Module):
    def __init__(self,
                 in_channels,
                 norm_args=None,
                 act_args=None,
                 aggr_args={'feature_type': 'dp_fj', "reduction": 'max'},
                 group_args={'NAME': 'ballquery'},
                 conv_args=None,
                 expansion=1,
                 use_res=True,
                 num_posconvs=2,
                 less_act=False,
                 gamma=16,
                 num_gp=16,
                 tau_delta=1,
                 **kwargs
                 ):
        super().__init__()
        
        self.gamma = gamma
        self.num_gp = num_gp
        self.tau_delta = tau_delta
        
        self.use_res = use_res
        self.feature_type = aggr_args['feature_type']
        channels = [in_channels, in_channels,in_channels]
        
        channels[0] = CHANNEL_MAP[self.feature_type](channels[0])
        convs = []
        gconvs = []

        self.attn_local = create_convblock2d(channels[0], channels[-1],
                                norm_args=norm_args,
                                act_args=None,
                                **conv_args)
        
        for i in range(len(channels) - 1):
            convs.append(create_convblock2d(channels[i], channels[i + 1],
                                    norm_args=norm_args,
                                    act_args=None if i == len(channels) - 2 else act_args,
                                    **conv_args)
                        )
        
        self.convs = nn.Sequential(*convs)
        self.grouper = create_grouper(group_args)
        
        
        mid_channels = int(in_channels * expansion)
        if num_posconvs < 1:
            channels = []
        elif num_posconvs == 1:
            channels = [in_channels, in_channels]
        else:
            channels = [in_channels, mid_channels, in_channels]
        ffn = []
        # point wise after depth wise conv (without last layer)
        for i in range(len(channels) - 1):
            ffn.append(create_convblock1d(channels[i], channels[i + 1],
                                             norm_args=norm_args,
                                             act_args=act_args if
                                             (i != len(channels) - 2) and not less_act else None,
                                             **conv_args)
                          )
        self.ffn = nn.Sequential(*ffn)
        self.act = create_act(act_args)

        self.alpha=nn.Parameter(torch.zeros((1,), dtype=torch.float32)) 
        
    def forward(self, pf):
        p,f = pf
        
        identity = f
        dp, fj = self.grouper(p, p, f)
        fj = get_aggregation_features(p, dp, f, fj, self.feature_type)
        
        f = new_attention(self.convs(fj), attn = self.attn_local(fj))

        f = self.act(f+identity)
        identity=f
        
        f = self.ffn(f)
        if f.shape[-1] == identity.shape[-1] and self.use_res:
            f += identity
        f = self.act(f)
        
        return p,f

@MODELS.register_module()
class SPoTrPartDecoder(nn.Module):
    def __init__(self,
                 encoder_channel_list: List[int],
                 decoder_layers: int = 5,
                 decoder_blocks: List[int] = [1, 1, 1, 1],
                 decoder_strides: List[int] = [4, 4, 4, 4],
                 act_args: str = 'relu',
                 num_classes: int = 50,
                 **kwargs
                 ):
        super().__init__()
        self.decoder_layers = decoder_layers
        self.in_channels = encoder_channel_list[-1]
        skip_channels = encoder_channel_list[:-1]
        fp_channels = encoder_channel_list[:-1]
        
        # the following is for decoder blocks
        self.conv_args = kwargs.get('conv_args', {'kernel_size': 1})
        radius_scaling = kwargs.get('radius_scaling', 2)
        nsample_scaling = kwargs.get('nsample_scaling', 1)
        block = kwargs.get('block', 'LPAMLP')
        if isinstance(block, str):
            block = eval(block)
        self.blocks = decoder_blocks
        self.strides = decoder_strides
        self.norm_args = kwargs.get('norm_args', {'norm': 'bn'}) 
        self.act_args = kwargs.get('act_args', {'act': 'relu'}) 
        self.expansion = kwargs.get('expansion', 4)
        radius = kwargs.get('radius', 0.1)
        nsample = kwargs.get('nsample', 16)
        self.radii = self._to_full_list(radius, radius_scaling)
        self.nsample = self._to_full_list(nsample, nsample_scaling)
        self.num_classes = num_classes
        self.use_res = kwargs.get('use_res', True)
        group_args = kwargs.get('group_args', {'NAME': 'ballquery'})
        self.aggr_args = kwargs.get('aggr_args', 
                                    {'feature_type': 'dp_fj', "reduction": 'max'}
                                    )  

        # global features
        self.global_conv2 = nn.Sequential(
            create_convblock1d(encoder_channel_list[-1] , 128,
                                norm_args=None,
                                act_args=act_args))
        self.global_conv1 = nn.Sequential(
            create_convblock1d(encoder_channel_list[-2] , 64,
                                norm_args=None,
                                act_args=act_args))
        skip_channels[0] += 64 + 128 + 50  # shape categories labels


        n_decoder_stages = len(fp_channels)
        decoder = [[] for _ in range(n_decoder_stages)]
        for i in range(-1, -n_decoder_stages - 1, -1):
            group_args.radius = self.radii[i]
            group_args.nsample = self.nsample[i]
            decoder[i] = self._make_dec(
                skip_channels[i], fp_channels[i], group_args=group_args, block=block, blocks=self.blocks[i])

        self.decoder = nn.Sequential(*decoder)
        self.out_channels = fp_channels[-n_decoder_stages]

    def _make_dec(self, skip_channels, fp_channels, group_args=None, block=None, blocks=1):
        layers = []
        radii = group_args.radius
        nsample = group_args.nsample
        mlp = [skip_channels + self.in_channels] + \
              [fp_channels] * self.decoder_layers
        layers.append(FeaturePropogation(mlp, act_args=self.act_args))
        self.in_channels = fp_channels
        for i in range(1, blocks):
            group_args.radius = radii[i]
            group_args.nsample = nsample[i]
            layers.append(block(self.in_channels,
                                aggr_args=self.aggr_args,
                                norm_args=self.norm_args, act_args=self.act_args, group_args=group_args,
                                conv_args=self.conv_args, expansion=self.expansion,
                                use_res=self.use_res
                                ))
        return nn.Sequential(*layers)

    def _to_full_list(self, param, param_scaling=1):
        # param can be: radius, nsample
        param_list = []
        if isinstance(param, List):
            # make param a full list
            for i, value in enumerate(param):
                value = [value] if not isinstance(value, List) else value
                if len(value) != self.blocks[i]:
                    value += [value[-1]] * (self.blocks[i] - len(value))
                param_list.append(value)
        else:  # radius is a scalar (in this case, only initial raidus is provide), then create a list (radius for each block)
            for i, stride in enumerate(self.strides):
                if stride == 1:
                    param_list.append([param] * self.blocks[i])
                else:
                    param_list.append(
                        [param] + [param * param_scaling] * (self.blocks[i] - 1))
                    param *= param_scaling
        return param_list
    
    
    def get_num_layers(self):
        # Global convolutions
        global_layers = len(self.global_conv1) + len(self.global_conv2)

        # Decoder layers
        decoder_layers = 0
        for i in range(len(self.blocks)):
            # Each decoder block contains `decoder_layers` layers
            decoder_layers += self.decoder_layers * self.blocks[i]

        # Total number of layers in the decoder
        total_layers = global_layers + decoder_layers
        return total_layers

    def forward(self, p, f, cls_label):
        B, N = p[0].shape[0:2]

        emb1 = self.global_conv1(f[-2])
        emb1 = emb1.max(dim=-1, keepdim=True)[0]  # bs, 64, 1
        emb2 = self.global_conv2(f[-1])
        emb2 = emb2.max(dim=-1, keepdim=True)[0]  # bs, 128, 1
        cls_one_hot = torch.zeros((B, self.num_classes), device=p[0].device)
        cls_one_hot = cls_one_hot.scatter_(1, cls_label, 1).unsqueeze(-1)
        cls_one_hot = torch.cat((emb1, emb2, cls_one_hot), dim=1)
        cls_one_hot = cls_one_hot.expand(-1, -1, N)

        for i in range(-1, -len(self.decoder), -1):
            f[i - 1] = self.decoder[i][1:](
                [p[i-1], self.decoder[i][0]([p[i - 1], f[i - 1]], [p[i], f[i]])])[1]

        f[-len(self.decoder) - 1] = self.decoder[0][1:](
            [p[0], self.decoder[0][0]([p[0], torch.cat([cls_one_hot, f[0]], 1)], [p[1], f[1]])])[1]

        return f[-len(self.decoder) - 1]