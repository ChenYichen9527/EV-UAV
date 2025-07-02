import os
from configs.configs import cfg
import torch
import numpy as np
from dataset.basedataset import BaseDataLoader

class EvUAV(BaseDataLoader):
    def __init__(self, configs, mode='train'):
        super().__init__(configs)

        self.mode = mode
        self.root = os.path.join(self.root,mode)
        self.file_list = os.listdir(self.root)

    def __getitem__(self, idx):
        events = np.load(os.path.join(self.root,self.file_list[idx]))
        evs_norm,ev_loc,seg_label,idx= events['evs_norm'][:,0:4],events['ev_loc'],events['evs_norm'][:,4],events['evs_norm'][:,5]


        if self.mode=='train':
            num_events = ev_loc.shape[0]
            if num_events >= cfg.max_events_num:
                dowmsample_idx = np.random.choice(num_events,cfg.max_events_num,replace=False)
                ev_loc = ev_loc[dowmsample_idx]
                evs_norm=evs_norm[dowmsample_idx]
                seg_label = seg_label[dowmsample_idx]
                idx = idx[dowmsample_idx]
                print('downsample')

        out={}
        out['ev_loc']=ev_loc
        out['evs_norm']=evs_norm
        out['seg_label']=seg_label
        out['idx'] = idx

        return out


    def __len__(self):
        return len(self.file_list)


