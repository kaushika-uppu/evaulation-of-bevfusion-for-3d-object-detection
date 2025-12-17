import os
os.environ['MPLBACKEND'] = 'Agg'

import time
import statistics

import torch

from mmengine.config import Config
from mmengine.registry import init_default_scope

from mmdet3d.apis import init_model
from mmdet3d.registry import DATASETS
from mmdet3d.structures import Det3DDataSample


def build_nuscenes_val_dataset(cfg):
    """Build the NuScenes val/test dataset from the BEVFusion config."""
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
    checkpoint_file = (
        "checkpoints/"
        "bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth"
    )

    print("Loading config...")
    cfg = Config.fromfile(config_file)

    print("Building NuScenes dataset (val/test split)...")
    dataset = build_nuscenes_val_dataset(cfg)
    print(f"Dataset size: {len(dataset)} samples")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    print("Initializing BEVFusion model...")
    model = init_model(config_file, checkpoint_file, device=device)
    model.eval()

    # ---- Latency measurement ----
    warmup_iters = 5
    measure_iters = 30
    times_ms = []

    print("Starting latency measurement (model.predict on single samples)...")

    for idx in range(len(dataset)):
        if idx >= warmup_iters + measure_iters:
            break

        sample = dataset[idx]
        inputs = sample['inputs']
        data_sample = sample['data_samples']

        # ---------- POINTS ----------
        # dataset gives a single Tensor [N, 5]; voxelize() expects list[Tensor]
        points = inputs['points']
        if isinstance(points, torch.Tensor):
            points_list = [points.to(device)]
        elif isinstance(points, list):
            points_list = [p.to(device) for p in points]
        else:
            raise TypeError(f"Unexpected type for points: {type(points)}")

        # ---------- IMAGES ----------
        imgs = inputs['img']

        # We want a single tensor: [B, num_cams, C, H, W]
        if isinstance(imgs, list):
            # Common case: list of per-camera tensors [C, H, W]
            if len(imgs) > 0 and isinstance(imgs[0], torch.Tensor):
                if imgs[0].dim() == 3:
                    # [num_cams, C, H, W]
                    imgs_tensor = torch.stack([im.to(device) for im in imgs], dim=0)
                elif imgs[0].dim() == 4:
                    # e.g. imgs[0] is already [num_cams, C, H, W]
                    imgs_tensor = imgs[0].to(device)
                else:
                    raise ValueError(f"Unexpected img tensor dim: {imgs[0].dim()}")
            else:
                raise TypeError(f"Unexpected list content for imgs: {type(imgs[0]) if imgs else 'empty'}")
        elif isinstance(imgs, torch.Tensor):
            imgs_tensor = imgs.to(device)
        else:
            raise TypeError(f"Unexpected type for imgs: {type(imgs)}")

        # Add batch dim if needed -> [1, num_cams, C, H, W]
        if imgs_tensor.dim() == 4:
            imgs_tensor = imgs_tensor.unsqueeze(0)

        batch_inputs_dict = {
            'points': points_list,   # list[Tensor]
            'imgs': imgs_tensor      # Tensor [B, num_cams, C, H, W]
        }

        # ---------- DATA SAMPLES ----------
        if isinstance(data_sample, Det3DDataSample):
            batch_data_samples = [data_sample.to(device)]
        elif isinstance(data_sample, list):
            batch_data_samples = [ds.to(device) for ds in data_sample]
        else:
            raise TypeError(f"Unexpected type for data_samples: {type(data_sample)}")

        # ---- timing ----
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.time()
        with torch.no_grad():
            _ = model.predict(batch_inputs_dict, batch_data_samples)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        dt_ms = (time.time() - t0) * 1000.0

        if idx < warmup_iters:
            print(f"[Warmup]   Iter {idx + 1}: {dt_ms:.2f} ms")
        else:
            times_ms.append(dt_ms)
            print(f"[Measured] Iter {idx - warmup_iters + 1}: {dt_ms:.2f} ms")

    if times_ms:
        mean_ms = statistics.mean(times_ms)
        med_ms = statistics.median(times_ms)
        print("\n====== BEVFusion Latency on NuScenes (Colab GPU) ======")
        print(f"Number of measured samples: {len(times_ms)}")
        print(f"Mean latency:   {mean_ms:.2f} ms")
        print(f"Median latency: {med_ms:.2f} ms")
        print("======================================================")


if __name__ == "__main__":
    main()
