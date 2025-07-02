import torch
import numpy as np
import torch.nn as nn
import spconv.pytorch as spconv
from spconv.pytorch import  SparseModule
from torch.autograd import Function

import functools
from collections import OrderedDict
from lib.hais_ops import HAIS_OP


class BaseModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        input_c = cfg.input_c
        width = cfg.width

class Shortcut(nn.Module):
    def __init__(self, in_channels, out_channels,norm_fn, stride=1):
        super(Shortcut, self).__init__()
        self.conv = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=1, padding=1, bias=False),
            norm_fn(out_channels)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class SparseAgvPool(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x):
        x_features = x.features
        agv = torch.mean(x_features,dim=0)
        return agv


class SEModule(nn.Module):
    def __init__(self, channel, reduction):
        super().__init__()
        self.avg_pool = SparseAgvPool()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        se = self.avg_pool(x)
        se = self.fc(se)
        x = x.replace_feature(se * x.features)
        return x


class GDConv(SparseModule):
    def __init__(self, out_channels, norm_fn, stride=1, dilations=[1,2,3,4],indice_key=None):
        super().__init__()
        num_splits = len(dilations)
        assert (out_channels % num_splits == 0)
        temp = out_channels // num_splits
        convs = []
        for d in dilations:
            convs.append(spconv.SubMConv3d(temp, temp, kernel_size=3, padding=d, dilation=d,stride=stride))
        self.convs = nn.ModuleList(convs)
        self.num_splits=num_splits
        self.temp = temp

    def forward(self,x):
        res = []
        for i in range(self.num_splits):
            features_split = x.features[:,i*self.temp:(i+1)*self.temp]
            x_split = spconv.SparseConvTensor(features_split,x.indices,x.spatial_shape,x.batch_size)
            res.append(self.convs[i](x_split).features)
        x_features = torch.cat(res,dim=1)
        x = x.replace_feature(x_features)

        return x



class GDBlock(SparseModule):
    def __init__(self, in_channels, out_channels,kernel_size, stride, norm_fn, dilations=[1,2,3,4],indice_key=None,bias=False,ad_channels=16):
        """

        :rtype: object
        """
        super().__init__()

        self.shortcut = Shortcut(in_channels,out_channels,norm_fn)
        add_channel = ad_channels
        self.pwconv=spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels+add_channel, kernel_size=1, padding=1, bias=bias),
            norm_fn(out_channels+add_channel),
            nn.ReLU(),
        )

        self.gdconv=spconv.SparseSequential(
            GDConv(out_channels+add_channel,norm_fn, stride, dilations=dilations,indice_key=indice_key),
            norm_fn(out_channels+add_channel),
            nn.ReLU(),
        )

        self.se = SEModule(out_channels+add_channel,reduction=2)

        self.conv3=spconv.SparseSequential(
            spconv.SubMConv3d(out_channels+add_channel, out_channels, kernel_size=1, padding=1, bias=False),
            norm_fn(out_channels),
        )

        self.act = spconv.SparseSequential(nn.ReLU())


    def forward(self, input):
        identity = spconv.SparseConvTensor(input.features, input.indices, input.spatial_shape, input.batch_size)
        identity = self.shortcut(identity)
        x = self.pwconv(input)
        x = self.gdconv(x)
        x = self.se(x)
        x = self.conv3(x)

        x = x.replace_feature(x.features+identity.features)
        x = self.act(x)

        return x


def Downsample_block(in_channels, out_channels, kernel_size,norm_fn, stride=2, padding=1,bias=False,indice_key=None ):
    m = spconv.SparseSequential(
        spconv.SparseConv3d(in_channels, out_channels, [3,3,5], stride=[2,2,4], padding=padding,bias=False,indice_key=indice_key),
        norm_fn(out_channels),
        nn.ReLU(),
    )
    return m
