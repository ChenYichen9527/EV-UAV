import torch
import numpy as np
from torch.autograd import Function
from lib.hais_ops import HAIS_OP
import spconv.pytorch as spconv

class Voxelization_Idx(Function):
    @staticmethod
    def forward(ctx, coords, batchsize, mode=4):
        '''
        :param ctx:
        :param coords:  long (N, dimension + 1) or (N, dimension) dimension = 3
        :param batchsize
        :param mode: int 4=mean
        :param dimension: int
        :return: output_coords:  long (M, dimension + 1) (M <= N)
        :return: output_map: int M * (maxActive + 1)
        :return: input_map: int N
        '''
        assert coords.is_contiguous()
        N = coords.size(0)
        output_coords = coords.new()

        input_map = torch.IntTensor(N).zero_()
        output_map = input_map.new()

        HAIS_OP.voxelize_idx(coords, output_coords, input_map, output_map, batchsize, mode)
        return output_coords, input_map, output_map

    @staticmethod
    def backward(ctx, a=None, b=None, c=None):
        return None
voxelization_idx = Voxelization_Idx.apply

class Voxelization(Function):
    @staticmethod
    def forward(ctx, feats, map_rule, mode=4):
        '''
        :param ctx:
        :param map_rule: cuda int M * (maxActive + 1)
        :param feats: cuda float N * C
        :return: output_feats: cuda float M * C
        '''
        assert map_rule.is_contiguous()
        assert feats.is_contiguous()
        N, C = feats.size()
        M = map_rule.size(0)
        maxActive = map_rule.size(1) - 1

        output_feats = torch.cuda.FloatTensor(M, C).zero_()

        ctx.for_backwards = (map_rule, mode, maxActive, N)

        HAIS_OP.voxelize_fp(feats, output_feats, map_rule, mode, M, maxActive, C)
        return output_feats

    @staticmethod
    def backward(ctx, d_output_feats):
        map_rule, mode, maxActive, N = ctx.for_backwards
        M, C = d_output_feats.size()

        d_feats = torch.cuda.FloatTensor(N, C).zero_()

        HAIS_OP.voxelize_bp(d_output_feats.contiguous(), d_feats, map_rule, mode, M, maxActive, C)
        return d_feats, None, None
voxelization = Voxelization.apply




class BaseDataLoader(torch.utils.data.Dataset):
    """
    Base class for dataloader.
    """

    def __init__(self, configs):
        self.configs = configs
        self.root = configs.root
        self.whole_t = configs.whole_t
        self.res = configs.res

    @staticmethod
    def custom_collate(batch):
        batch_size = len(batch)
        loc_batches=[]
        feature_batches=[]
        seg_label_batches=[]
        idx_label_batches = []

        for i,ev in enumerate(batch):
            ev_loc = ev['ev_loc']
            loc = np.hstack((i * np.ones((ev_loc.shape[0], 1)), ev_loc))
            loc_batches.append(loc)

            feature = ev['evs_norm'][:,0:4]
            feature_batches.append(feature)

            seg_label = ev['seg_label']

            seg_label_batches.append(seg_label)

            idx_label =ev['idx']
            idx_label_batches.append(idx_label)





        locs_batches = np.concatenate(loc_batches, axis=0)
        seg_label_batches = np.concatenate(seg_label_batches, axis=0)
        idx_label_batches = np.concatenate(idx_label_batches, axis=0)


        locs_batches = torch.from_numpy(locs_batches).to(torch.int64).contiguous()
        voxel_locs, p2v_map, v2p_map = voxelization_idx(locs_batches, batch_size, 4)

        feature_batches = torch.from_numpy(np.concatenate(feature_batches, axis=0)).contiguous()
        feature_batches =feature_batches.float()
        voxel_feats = voxelization(feature_batches.cuda(), v2p_map.cuda(), 4)
        voxel_feats = voxel_feats.cuda()


        spatial_shape = np.array([11*32,9*32,256*32])
        voxel_ev = spconv.SparseConvTensor(voxel_feats, voxel_locs.int().cuda(), spatial_shape, batch_size)

        output = {}

        output['voxel_ev'] = voxel_ev
        output['seg_label'] = torch.from_numpy(seg_label_batches)
        output['p2v_map'] = p2v_map
        output['locs'] = locs_batches
        output['idx_label'] = idx_label_batches

        return output



