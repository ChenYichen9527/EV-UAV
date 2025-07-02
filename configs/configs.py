import argparse
import yaml
import os

def get_parser():
    parser = argparse.ArgumentParser(description='Event Point Cloud Segmentation')
    parser.add_argument('--config', default='/media/admin1/123/ev-spsegnet_github/configs/evisseg_evuav.yaml',type=str, help='path to config file')

    args_cfg = parser.parse_args()
    assert args_cfg.config is not None
    with open(args_cfg.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.CLoader)
    for key in config:
        for k, v in config[key].items():
            setattr(args_cfg, k, v)
    return args_cfg

cfg = get_parser()

