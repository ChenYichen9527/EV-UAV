import os
from configs.configs import cfg
import torch
import torch.nn as nn
import numpy as np
from dataset.ev_uav import EvUAV
import random
from model.evspsegnet import evspsegnet
from utils.stcloss import STCLoss

import torch.optim as optim
import mlflow
import tqdm
from utils.eval import evalute

def setup(seed):
    seed_n = seed
    print('random seed:' + str(seed_n))
    g = torch.Generator()
    g.manual_seed(seed_n)
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)
    torch.cuda.manual_seed_all(seed_n)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    os.environ['PYTHONHASHSEED'] = str(seed_n)

if __name__ == '__main__':

    seed=37
    setup(seed)
    device = "cuda:0"

    net = evspsegnet(cfg).train()
    net.cuda()

    dataset = EvUAV(cfg,mode='train')
    train_sampler = torch.utils.data.sampler.RandomSampler(list(range(len(dataset))))
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, collate_fn=dataset.custom_collate, sampler=train_sampler)

    stc_criterion = STCLoss(k=cfg.k,t=cfg.t,cfg=cfg).cuda()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_loss = 1e5
    best_iou=0

    #for val
    val_dataset = EvUAV(cfg, mode='val')
    val_dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size,collate_fn=dataset.custom_collate)
    evaluter = evalute(cfg)

    # mlflow
    mlflow.set_experiment('train')
    mlflow.start_run(run_name='train')

    for epoch in range(cfg.epochs):
        pbar = tqdm.tqdm(total=len(train_dataloader), unit="Batch", unit_scale=True,
                         desc="Epoch: {}".format(epoch),position=0,leave=True)

        for ev in train_dataloader:
            x = ev['voxel_ev']
            label = ev['seg_label'].float().cuda()
            p2v_map = ev['p2v_map'].long().cuda()
            ev_locs = ev['locs'].float().requires_grad_()

            preds,voxel = net(x)

            loss = stc_criterion(voxel, p2v_map, preds, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss.item())
            pbar.update(1)

            with torch.no_grad():
                mlflow.log_metric('loss', loss.item())
                if loss.item()<best_loss:
                    torch.save(net.state_dict(),cfg.model_save_root+'/best_loss_seed{}.pt'.format(seed))
                    best_loss = loss.item()
            torch.cuda.empty_cache()

        scheduler.step()

        with torch.no_grad():
            if epoch>=40:
                for sample, ev in enumerate(val_dataloader):
                    x = ev['voxel_ev']
                    label = ev['seg_label'].float().cuda()
                    p2v_map = ev['p2v_map'].long().cuda()
                    ev_locs = ev['locs'].float().requires_grad_()
                    idx = ev['idx_label']
                    ts = ev_locs[:, 3]

                    preds, voxel = net(x)
                    preds = preds[p2v_map].squeeze().cpu()

                    evaluter.matches[str(sample)] = {}
                    evaluter.matches[str(sample)]['seg_pred'] = preds
                    evaluter.matches[str(sample)]['seg_gt'] = label
                iou = evaluter.evaluate_semantic_segmantation_miou()

                if iou.item() > best_iou:
                    torch.save(net.state_dict(), cfg.model_save_root + '/best_iou_seed{}.pt'.format(seed))
                    best_loss = loss.item()
