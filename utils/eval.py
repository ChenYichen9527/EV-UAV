import os.path
import torch
import cv2
import numpy as np
import pandas as pd
import tqdm


class evalute():
    def __init__(self, cfg):
        self.matches = {}
        self.data = pd.DataFrame()

        if cfg.roc:
            self.pd_detT = cfg.pd_detT
            self.correct_thresh = cfg.correct_thresh
            self.whole_ev_num = 0  # 总共的事件点的数量
            self.obj_num = 0  # gt中目标的数量
            self.frame_num = 0  # 等效帧的数量
            self.correct_num = 0  # 不同置信度阈值下的正确预测的点的数量
            self.false_num = 0  # 不同置信度阈值下的虚警的点的数量

    def roc_update(self, ts, preds, idx, label, ev_locs, thresh=0.9):
        self.whole_ev_num += preds.shape[0]
        self.frame_num += int((ts.max() - ts.min()) / self.pd_detT)
        for i in range(int((ts.max() - ts.min()) / self.pd_detT + 1)):
            t_range = (ts > i * self.pd_detT) * (ts < (i + 1) * self.pd_detT)
            idx_frame, preds_frame, label_frame, ev_locs_frame = idx[t_range], preds[t_range], label[t_range], \
                ev_locs[:, 1:4][t_range]
            preds_frame_ori = preds_frame.clone()
            idx_list_frame = set(idx_frame)
            false_mask = np.zeros((260, 346), dtype=np.uint8)
            preds_frame[preds_frame_ori >= thresh] = 1
            preds_frame[preds_frame_ori < thresh] = 0

            # 计算检测Pd
            for idx_i in idx_list_frame:
                if idx_i != 0:  # 目标
                    self.obj_num += 1
                    preds_frame_i = preds_frame[idx_frame == idx_i]
                    label_frame_i = label_frame[idx_frame == idx_i]
                    num_correct_frame = (preds_frame_i == label_frame_i).sum()
                    if num_correct_frame / label_frame_i.sum() >= self.correct_thresh:  # 如果目标占比大于阈值，则认为检测准确
                        self.correct_num += 1

            # 计算虚警Fa
            false_ev = ev_locs_frame[(label_frame == 0) * (preds_frame == 1)]
            for ii in range(false_ev.shape[0]):
                false_mask[int(false_ev[:, 1][ii]), int(false_ev[:, 0][ii])] += 1  # 将虚警点映射为帧
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(false_mask, connectivity=8,
                                                                                        ltype=cv2.CV_32S)
                self.false_num += (num_labels - 1)

    def cal_roc(self):
        pd = self.correct_num / self.obj_num
        fa = self.false_num / (self.frame_num * 346 * 260)
        return pd, fa

    def evaluate_semantic_segmantation_miou(self, thresh=0.9):
        seg_gt_list = []
        seg_pred_list = []
        for k, v in self.matches.items():
            seg_gt_list.append(v['seg_gt'])
            seg_pred_list.append(v['seg_pred'])
        seg_gt_all = torch.cat(seg_gt_list, dim=0).cpu()
        seg_pred_all = torch.cat(seg_pred_list, dim=0).cpu()
        seg_pred_all[seg_pred_all >= thresh] = 1
        seg_pred_all[seg_pred_all < thresh] = 0
        assert seg_gt_all.shape == seg_pred_all.shape
        iou_list = []
        for _index in seg_gt_all.unique():
            if _index == 1:
                intersection = ((seg_gt_all == _index) & (seg_pred_all == _index)).sum()
                union = ((seg_gt_all == _index) | (seg_pred_all == _index)).sum()
                iou = intersection.float() / union
                iou_list.append(iou)
        iou_tensor = torch.tensor(iou_list)
        miou = iou_tensor.mean()
        return miou

    def evaluate_semantic_segmantation_accuracy(self, thresh=0.9):
        seg_gt_list = []
        seg_pred_list = []
        for k, v in self.matches.items():
            seg_gt_list.append(v['seg_gt'])
            seg_pred_list.append(v['seg_pred'])
        seg_gt_all = torch.cat(seg_gt_list, dim=0).cuda()
        seg_pred_all = torch.cat(seg_pred_list, dim=0).cuda()
        seg_pred_all[seg_pred_all >= thresh] = 1
        seg_pred_all[seg_pred_all < thresh] = 0
        assert seg_gt_all.shape == seg_pred_all.shape
        correct = (seg_gt_all[seg_gt_all == 1] == seg_pred_all[seg_gt_all == 1]).sum()
        whole = (seg_gt_all == 1).sum()
        seg_accuracy = correct.float() / whole.float()
        return seg_accuracy
