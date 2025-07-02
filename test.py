import torch
from configs.configs import cfg
from dataset.ev_uav import EvUAV
from model.evspsegnet import evspsegnet
from utils.eval import evalute
import tqdm

if __name__ == '__main__':
    device = "cuda:0"

    net = evspsegnet(cfg).eval()
    net.cuda()

    dataset = EvUAV(cfg, mode='test')

    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size,collate_fn=dataset.custom_collate)

    net.load_state_dict(torch.load(cfg.model_path))
    print('dict load: ',cfg.model_path)


    pbar = tqdm.tqdm(total=len(test_dataloader), desc='video', unit='video',unit_scale=True,position=0, leave=True)

    evaluter = evalute(cfg)

    for sample,ev in enumerate(test_dataloader):
        with torch.no_grad():
            x = ev['voxel_ev']
            label = ev['seg_label'].float().cuda()
            p2v_map = ev['p2v_map'].long().cuda()
            ev_locs = ev['locs'].float().requires_grad_()
            idx = ev['idx_label']
            ts = ev_locs[:,3]

            preds, voxel = net(x)
            preds = preds[p2v_map].squeeze().cpu()

            if cfg.eval:
                evaluter.matches[str(sample)] = {}
                evaluter.matches[str(sample)]['seg_pred']= preds
                evaluter.matches[str(sample)]['seg_gt'] = label
                if cfg.roc:
                    evaluter.roc_update(ts,preds,idx,label.cpu(),ev_locs)

        pbar.update(1)

    if cfg.eval:
        iou = evaluter.evaluate_semantic_segmantation_miou()
        seg_acc = evaluter.evaluate_semantic_segmantation_accuracy()
        if cfg.roc:
            pd, fa= evaluter.cal_roc()
        print('iou:{},seg_acc:{},pd:{},fa:{}'.format(iou,seg_acc,pd,fa))






