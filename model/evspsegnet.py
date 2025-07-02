import torch
import torch.nn as nn
import spconv.pytorch as spconv
import functools
from spconv.pytorch import functional as Fsp
from model.basemodel import GDBlock,Downsample_block
from lib.hais_ops import HAIS_OP
from configs.configs import cfg
import math
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class patch_attention(spconv.SparseModule,):
    def __init__(self, channel,in_sptial_size,att_sptial_size=(11,9,8),indice_key='pa1'):
        super(patch_attention, self).__init__()
        maxpool_num = math.log2(int(in_sptial_size[0]/att_sptial_size[0]))
        self.layers = nn.ModuleList()
        for i in range(int(maxpool_num)):
            self.layers.append(spconv.SparseMaxPool3d((2,2,4),stride=(2,2,4),indice_key=indice_key+str(i)))

        self.multihead_attn=nn.MultiheadAttention(embed_dim=channel,num_heads=1)
        self.inver_layer=nn.ModuleList()
        for i in range(int(maxpool_num)):
            self.inver_layer.append(spconv.SparseInverseConv3d(channel, channel, (2,2,4), indice_key=indice_key+str(int(maxpool_num-1)-i), bias=False,))
        self.conv = spconv.SubMConv3d(channel, channel, 1, stride=1, padding=1, bias=False,)

    def forward(self, x):
        identity = x.features
        for m in self.layers:
            x=m(x)
        x_features = x.features.unsqueeze(0)
        attn_output, attn_weight = self.multihead_attn(query=x_features, key=x_features, value=x_features)# attn_output的形状是(序列长度, 批大小, 特征数量)
        attn_output = attn_output.squeeze(0)
        x = x.replace_feature(attn_output)
        for m in self.inver_layer:
            x=m(x)
        x = x.replace_feature(x.features + identity)
        x = self.conv(x)

        return x


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None, algo = None,dilations=[1,2,3,4],ad_channels=16):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key,algo=algo)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key,algo=algo)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False,algo=algo)
    elif conv_type == 'gd':
        conv = GDBlock(in_channels, out_channels, kernel_size,stride,norm_fn,dilations=dilations, indice_key=indice_key, bias=False,ad_channels=ad_channels)

    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, indice_key=None, norm_fn=None):
        super(SparseBasicBlock, self).__init__()
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x.features

        assert x.features.dim() == 2, 'x.features.dim()=%d' % x.features.dim()

        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)
        out = out.replace_feature( self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out.replace_feature( out.features + identity)
        out = out.replace_feature(self.relu(out.features))

        return out

class evspsegnet(nn.Module):
    def __init__(self,cfg):
        super().__init__()

        input_channels = cfg.input_channel
        width=cfg.width

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, width, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(width),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(width, width, 3, norm_fn=norm_fn, padding=1, indice_key='subm1',conv_type='gd'),
        )

        self.conv2 = spconv.SparseSequential(
            block(width, 2*width, 3, norm_fn=norm_fn, stride=[2,2,4], padding=1, indice_key='spconv2', conv_type='spconv'),
            block(2*width, 2*width, 3, norm_fn=norm_fn, padding=1, indice_key='subm2', conv_type='gd',ad_channels=16),
        )
        self.pa2 = patch_attention(2*width, (176, 144, 2048), indice_key='pa2')

        self.conv3 = spconv.SparseSequential(

            block(2*width, 4*width, 3, norm_fn=norm_fn, stride=[2,2,4], padding=1, indice_key='spconv3', conv_type='spconv'),
            block(4*width, 4*width, 3, norm_fn=norm_fn, padding=1, indice_key='subm3', conv_type='gd',ad_channels=8),
        )
        self.pa3 = patch_attention(4*width, (88, 72, 512), indice_key='pa3')

        self.conv4 = spconv.SparseSequential(

            block(4*width, 4*width, 3, norm_fn=norm_fn, stride=[2,2,4], padding=1, indice_key='spconv4', conv_type='spconv'),
            block(4*width, 4*width, 3, norm_fn=norm_fn, padding=1, indice_key='subm4', conv_type='gd',ad_channels=0),
        )
        self.pa4 = patch_attention(4*width, (44, 36, 256), indice_key='pa4')

        # decoder
        self.conv_up_t4 = SparseBasicBlock(4*width, 4*width, indice_key='subm4', norm_fn=norm_fn)
        self.conv_up_m4 = block(8*width, 4*width, 3, norm_fn=norm_fn, padding=1, indice_key='subm4',conv_type='gd')
        self.inv_conv4 = block(4*width, 4*width, 3, norm_fn=norm_fn, indice_key='spconv4', conv_type='inverseconv')


        self.conv_up_t3 = SparseBasicBlock(4*width, 4*width, indice_key='subm3', norm_fn=norm_fn)
        self.conv_up_m3 = block(8*width, 4*width, 3, norm_fn=norm_fn, padding=1, indice_key='subm3',conv_type='gd')
        self.inv_conv3 = block(4*width, 2*width, 3, norm_fn=norm_fn, indice_key='spconv3', conv_type='inverseconv')


        self.conv_up_t2 = SparseBasicBlock(2*width, 2*width, indice_key='subm2', norm_fn=norm_fn)
        self.conv_up_m2 = block(4*width, 2*width, 3, norm_fn=norm_fn, indice_key='subm2',conv_type='gd')
        self.inv_conv2 = block(2*width, width, 3, norm_fn=norm_fn, indice_key='spconv2', conv_type='inverseconv')


        self.conv_up_t1 = SparseBasicBlock(width, width, indice_key='subm1', norm_fn=norm_fn)
        self.conv_up_m1 = block(2*width, width, 3, norm_fn=norm_fn, indice_key='subm1',conv_type='gd')

        self.conv5 = spconv.SparseSequential(
            block(width, width, 3, norm_fn=norm_fn, padding=1, indice_key='subm1',conv_type='gd')
        )

        self.semantic_linear = nn.Sequential(
            nn.Linear(width, 1),
            nn.Sigmoid()
        )
    def UR_block_forward(self, x_lateral, x_bottom, conv_t, conv_m, conv_inv):
        x_trans = conv_t(x_lateral)
        x = x_trans
        x = x.replace_feature(torch.cat((x_bottom.features, x_trans.features), dim=1))
        x_m = conv_m(x)
        x = self.channel_reduction(x, x_m.features.shape[1])
        x = x.replace_feature(x_m.features + x.features)
        x = conv_inv(x)
        return x

    @staticmethod
    def channel_reduction(x, out_channels):
        """
        Args:
            x: x.features (N, C1)
            out_channels: C2

        Returns:

        """
        features = x.features
        n, in_channels = features.shape
        assert (in_channels % out_channels == 0) and (in_channels >= out_channels)

        x = x.replace_feature( features.view(n, out_channels, -1).sum(dim=2))
        return x


    def forward(self, input):
        x = self.conv_input(input)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv2 = self.pa2(x_conv2)
        x_conv3 = self.conv3(x_conv2)
        x_conv3 = self.pa3(x_conv3)
        x_conv4 = self.conv4(x_conv3)
        x_conv4 = self.pa4(x_conv4)

        x_up4 = self.UR_block_forward(x_conv4, x_conv4, self.conv_up_t4, self.conv_up_m4, self.inv_conv4)
        x_up3 = self.UR_block_forward(x_conv3, x_up4, self.conv_up_t3, self.conv_up_m3, self.inv_conv3)
        x_up2 = self.UR_block_forward(x_conv2, x_up3, self.conv_up_t2, self.conv_up_m2, self.inv_conv2)
        x_up1 = self.UR_block_forward(x_conv1, x_up2, self.conv_up_t1, self.conv_up_m1, self.conv5)

        output = self.semantic_linear(x_up1.features)
        voxel = x_up1.replace_feature(output)
        return output,voxel
