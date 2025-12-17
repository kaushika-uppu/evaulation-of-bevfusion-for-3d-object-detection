import os
os.environ['MPLBACKEND'] = 'Agg'

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmdet3d.registry import DATASETS

def build_nuscenes_val_dataset(cfg):
    cfg.data_root = "/content/nuscenes"
    if 'val_dataloader' in cfg:
        data_cfg = cfg.val_dataloader
    else:
        data_cfg = cfg.test_dataloader

    dataset_cfg = data_cfg.dataset
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))
    dataset = DATASETS.build(dataset_cfg)
    return dataset

def main():
    config_file = (
        "projects/BEVFusion/configs/"
        "bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py"
    )
    cfg = Config.fromfile(config_file)
    dataset = build_nuscenes_val_dataset(cfg)
    print("Dataset length:", len(dataset))

    sample = dataset[0]
    print("Keys in sample:", sample.keys())

    inputs = sample['inputs']
    print("Keys in inputs:", inputs.keys())

    # This is what BEVFusion expects to read
    points = inputs.get('points', None)
    print("\nType of points:", type(points))

    if isinstance(points, list):
        print("points is a list of length:", len(points))
        if len(points) > 0:
            print("type(points[0]):", type(points[0]))
            if isinstance(points[0], torch.Tensor):
                print("points[0].shape:", points[0].shape)
            else:
                try:
                    import numpy as np
                    if isinstance(points[0], np.ndarray):
                        print("points[0].shape (np):", points[0].shape)
                except ImportError:
                    pass
    elif isinstance(points, torch.Tensor):
        print("points.shape:", points.shape)
    else:
        print("points is neither list nor Tensor:", points)

if __name__ == "__main__":
    main()
