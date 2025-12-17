# BEVFusion 3D Object Detection and Latency Evaluation on nuScenes-mini

1. Introduction
This project implements and evaluates BEVFusion, a multi-sensor 3D object detection model that fuses LiDAR and camera inputs into a unified Bird's-Eye View (BEV) representation.  
The goal is to install MMDetection3D and BEVFusion on Google Colab, prepare the nuScenes-mini dataset, load the model, and measure inference latency on GPU.

---

2. Setup Overview
All experiments were performed in Google Colab using:

- Python 3.10  
- CUDA 12.4  
- GPU: NVIDIA L4  
- PyTorch 2.3.0  
- MMEngine 0.10.7  
- MMCV 2.2.0  
- MMDetection 3.3.0  
- MMDetection3D 1.4.0  

These versions were chosen for compatibility with BEVFusion and recent CUDA toolchains.

3. Dataset
We use **nuScenes v1.0-mini**, which contains:
- 10 scenes  
- 404 samples  
- LiDAR + 6 cameras  

Dataset preprocessing was done using the MMDetection3D script:

```bash
python tools/create_data.py nuscenes --root-path /content/nuscenes \
  --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0-mini


4. BEVFusion Model

It can be configured using : 
projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py

5. Latency Evaluation
scripts/bevfusion_latency_predict_nuscenes.py

6. Results: 

Mean Latency:296.94ms
Median Latency:297.11ms
FPS:3.36

7. Run the colab cells one after another to get the evaluation metrics. 

8. References:

MMDetection3D: https://github.com/open-mmlab/mmdetection3d

BEVFusion Paper: https://doi.org/10.48550/arxiv.2205.13542

nuScenes Dataset: https://www.nuscenes.org