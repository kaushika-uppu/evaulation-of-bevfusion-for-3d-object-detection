import os
os.environ['MPLBACKEND'] = 'Agg'  # avoid matplotlib backend issues

import torch
from mmdet3d.apis import init_model

print("Torch:", torch.__version__, "CUDA:", torch.cuda.is_available())

config_file = "projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py"

# Use the exact checkpoint filename you downloaded:
checkpoint_file = "checkpoints/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth"

# 🔧 IMPORTANT: use cuda:0 instead of cuda
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = init_model(config_file, checkpoint_file, device=device)
print("Model initialized OK!")
