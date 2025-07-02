import torch
import torch.nn as nn
from torch.autograd import Function
from lib.hais_ops import HAIS_OP
import numpy as np
import spconv.pytorch as spconv


class STCLoss(nn.Module):
    def __init__(self, k, t, cfg,weight_clip_eps=1e-5):
        super(STCLoss, self).__init__()
        self.k = k
        self.t = t
        self.vol = self.k * self.k * self.t
        self.cfg = cfg

        self.stc_conv = spconv.SubMConv3d(1, 1, kernel_size=[self.k, self.k, self.t], stride=1,
                                          padding=[int(self.k / 2), int(self.k / 2), int(self.t / 2)], bias=False)

        weights = self.stc_conv.weight.data
        weights.fill_(1)
        self.stc_conv.requires_grad_(False)

        self.eps = weight_clip_eps

    def forward(self, voxel, p2v_map, preds, label):
        stc_voxel = self.stc_conv(voxel)
        mean_stc = torch.mean(stc_voxel.features)
        stc_weights = torch.sigmoid(stc_voxel.features-mean_stc)
        stc_weights = stc_weights[p2v_map].squeeze().detach()
        preds = preds[p2v_map].squeeze()
        preds = torch.clamp(preds, 0, 1)

        pos_loss = -torch.log(preds + self.eps)
        neg_loss = -torch.log(1 - preds + self.eps)

        loss = (label * stc_weights * pos_loss) + ((1 - label) * (1 - stc_weights) * neg_loss)
        return loss.mean()






