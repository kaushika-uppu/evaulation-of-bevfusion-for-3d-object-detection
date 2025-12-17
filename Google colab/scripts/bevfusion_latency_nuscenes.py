import os
os.environ['MPLBACKEND'] = 'Agg'  # avoid matplotlib backend issues

import time
import statistics

import torch

from mmengine.config import Config
from mmengine.registry import init_default_scope

from mmdet3d.apis import init_model
from mmdet3d.registry import DATASETS


def build_nuscenes_val_dataset(cfg):
    """Build the NuScenes val/test dataset from the BEVFusion config."""
    # Make sure the config knows where our data root is
    cfg.data_root = "/content/nuscenes"

    # BEVFusion configs usually have val_dataloader & test_dataloader;
    # we'll prefer val if it exists, otherwise fall back to test.
    if 'val_dataloader' in cfg:
        data_cfg = cfg.val_dataloader
    else:
        data_cfg = cfg.test_dataloader

    dataset_cfg = data_cfg.dataset

    # Make sure the default scope is set so registries work
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
    warmup_iters = 5     # first N samples ignored for timing
    measure_iters = 30   # number of timed samples
    times_ms = []

    print("Starting latency measurement (direct dataset indexing, no DataLoader)...")

    with torch.no_grad():
        total = min(len(dataset), warmup_iters + measure_iters)
        for i in range(total):
            # Get one sample from the dataset
            sample = dataset[i]

            # 🔑 KEY FIX:
            # mmdet3d data_preprocessor expects:
            #   - a dict with 'inputs' and 'data_samples'
            #   - AND data_samples to be a *list* of Det3DDataSample
            batch = {
                'inputs': sample['inputs'],
                'data_samples': [sample['data_samples']],  # <-- wrap in list ✅
            }

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            t0 = time.time()
            _ = model.test_step(batch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            dt_ms = (time.time() - t0) * 1000.0

            if i < warmup_iters:
                print(f"[Warmup]   Iter {i + 1}: {dt_ms:.2f} ms")
            else:
                times_ms.append(dt_ms)
                print(f"[Measured] Iter {i - warmup_iters + 1}: {dt_ms:.2f} ms")

    if times_ms:
        mean_ms = statistics.mean(times_ms)
        med_ms = statistics.median(times_ms)
        print("\n====== BEVFusion Latency on NuScenes (Colab GPU) ======")
        print(f"Number of measured samples: {len(times_ms)}")
        print(f"Mean latency:   {mean_ms:.2f} ms")
        print(f"Median latency: {med_ms:.2f} ms")
        print("======================================================")
    else:
        print("No latency measurements were collected.")


if __name__ == "__main__":
    main()
