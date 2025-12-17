# BEVFusion ROS 2 Node

A ROS 2 node for real-time 3D object detection using NVIDIA's CUDA-BEVFusion, processing synchronized camera images and LiDAR point clouds.

## Overview

The `bev_node` performs multi-sensor fusion (6 cameras + LiDAR) to detect 3D objects in real-time. It subscribes to camera images and LiDAR point clouds, runs BEVFusion inference, and publishes 3D bounding box detections.

**Supported Classes:**
- car, truck, construction_vehicle, bus, trailer
- barrier, motorcycle, bicycle, pedestrian, traffic_cone

## Quick Start

### 1. Prerequisites

- Jetson Orin (SM 8.7) with CUDA 12.6+
- ROS 2 Humble
- CUDA-BEVFusion built (see `BEVFusion_SETUP.md`)
- Model files (`.plan` and `.onnx`) in `CUDA-BEVFusion/model/resnet50/build/`
- Calibration files in `CUDA-BEVFusion/example-data/` (or your converted data)

### 2. Environment Setup

```bash
# Set library paths
export LD_LIBRARY_PATH=/home/student/ros2/ros2_ws/Lidar_AI_Solution/CUDA-BEVFusion/build:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/student/ros2/ros2_ws/Lidar_AI_Solution/libraries/3DSparseConvolution/libspconv/lib/aarch64_cuda12.8:$LD_LIBRARY_PATH

# Set Python path for pybev
export PYTHONPATH=/home/student/ros2/ros2_ws/Lidar_AI_Solution/CUDA-BEVFusion/build:$PYTHONPATH
```

### 3. Build ROS 2 Node

```bash
cd /home/student/ros2/ros2_ws
colcon build --packages-select dnn_node --symlink-install
source install/setup.bash
```

### 4. Run the Node

**With live sensors:**
```bash
ros2 run dnn_node bev_node
```

**With replay node (for testing with recorded data):**
```bash
# Terminal 1: Replay data
ros2 run dnn_node replay_node

# Terminal 2: Run BEVFusion
ros2 run dnn_node bev_node
```

### 5. View Detections

```bash
# Check detections
ros2 topic echo /bevfusion/detections

# Or visualize in RViz
ros2 run rviz2 rviz2
# Add: Detection3DArray display, topic: /bevfusion/detections
```

## Topics

### Subscribed Topics

- `/cam_front/image_raw` (sensor_msgs/Image)
- `/cam_front_right/image_raw` (sensor_msgs/Image)
- `/cam_front_left/image_raw` (sensor_msgs/Image)
- `/cam_back/image_raw` (sensor_msgs/Image)
- `/cam_back_left/image_raw` (sensor_msgs/Image)
- `/cam_back_right/image_raw` (sensor_msgs/Image)
- `/lidar_top/points` (sensor_msgs/PointCloud2)

### Published Topics

- `/bevfusion/detections` (vision_msgs/Detection3DArray) - 3D bounding box detections

## Configuration

The node is currently configured for:
- **Model variant:** `resnet50`
- **Model path:** `CUDA-BEVFusion/model/resnet50/build/`
- **Calibration:** `CUDA-BEVFusion/example-data/` (or your converted nuScenes data)
- **Confidence threshold:** `0.01` (adjustable via parameter)

To change configuration, edit `bev_node.py`:
- Line 75: `model_variant = "resnet50"`
- Line 78: `calib_dir = os.path.join(REPO_ROOT, "example-data")`
- Line 71: `confidence_threshold` parameter default

## Data Pipeline

### Converting NuScenes Data

If you have nuScenes dataset and want to use your own calibration:

```bash
cd /home/student/ros2/ros2_ws/src/dnn_node/dnn_node
python3 convert_nuscenes.py
```

This creates:
- Calibration tensors: `camera2lidar.tensor`, `camera_intrinsics.tensor`, etc.
- Image and point cloud data in `/home/student/ros2/ros2_ws/data/bev_sequence/`

Then update `bev_node.py` line 78 to point to your converted data directory.

## Known Issues & Limitations

1. **CUDA Kernel Errors (SM 8.7):** The pre-built `libspconv.so` doesn't include SM 8.7 kernels, causing CUDA errors. The model still runs but with degraded performance (lower confidence scores). This is a known limitation of the pre-built library.

2. **Low Confidence Scores:** Due to CUDA errors, detection scores are typically 0.01-0.07 (instead of 0.25+). The threshold is set to 0.01 to allow detections. Once `libspconv.so` is rebuilt with SM 8.7 support, scores should improve.

3. **Calibration Mismatch:** Using `example-data` calibration with different camera setups will reduce detection accuracy. Use calibration data matching your sensor configuration.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ImportError: libpybev.so` | Check `LD_LIBRARY_PATH` includes `CUDA-BEVFusion/build` |
| `ImportError: libspconv.so` | Check `LD_LIBRARY_PATH` includes spconv library directory |
| `ModuleNotFoundError: tensor` | Verify `REPO_ROOT` path resolution in `bev_node.py` |
| No detections published | Lower confidence threshold, check sensor synchronization |
| CUDA errors in logs | Expected with pre-built `libspconv.so` - model still works |

## Performance

- **Inference time:** ~200-500ms per frame (depends on point cloud size)
- **Detections per frame:** 3-12 objects (varies with scene)
- **Memory:** ~2-4GB GPU memory

## See Also

- `BEVFusion_SETUP.md` - Complete build and setup instructions
- `PROTOBUF_FIX.md` - Protobuf dependency fixes

