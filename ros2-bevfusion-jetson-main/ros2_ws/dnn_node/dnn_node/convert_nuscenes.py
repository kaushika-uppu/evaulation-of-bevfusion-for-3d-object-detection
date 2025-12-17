import os
import sys
import numpy as np
import cv2
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

# Import tensor save/load utility (works with NumPy, no PyTorch needed)
sys.path.insert(0, '/root/ros2_ws/Lidar_AI_Solution/CUDA-BEVFusion/tool')
from tensor import save as save_tensor

# CONFIG
NUSC_ROOT = "/root/ros2_ws/data/nuscenes"  # Adjust if different
OUTPUT_DIR = "/root/ros2_ws/data/bev_sequence"
SCENE_IDX = 0  # Pick the first scene in the mini set

def get_sensor_transform(nusc, sample_data_token, target_frame='ego'):
    """Get transformation matrix from sensor to target frame."""
    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    
    # Rotation/Translation from Sensor to Ego
    r = Quaternion(cs_record['rotation']).rotation_matrix
    t = np.array(cs_record['translation'])
    
    # 4x4 Matrix
    tf = np.eye(4)
    tf[:3, :3] = r
    tf[:3, 3] = t
    return tf, cs_record

def main():
    if not os.path.exists(NUSC_ROOT):
        print(f"Error: NuScenes not found at {NUSC_ROOT}")
        return

    print("Loading NuScenes... (this takes time)")
    nusc = NuScenes(version='v1.0-mini', dataroot=NUSC_ROOT, verbose=True)
    
    scene = nusc.scene[SCENE_IDX]
    print(f"Converting Scene: {scene['name']}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- 1. EXTRACT STATIC CALIBRATION (From first frame) ---
    first_sample = nusc.get('sample', scene['first_sample_token'])
    lidar_token = first_sample['data']['LIDAR_TOP']
    
    # Lidar to Ego
    l2e, _ = get_sensor_transform(nusc, lidar_token)
    e2l = np.linalg.inv(l2e) # Ego to Lidar
    
    # Camera Intrinsics & Extrinsics
    cams = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    
    intrinsics_list = []
    c2l_list = [] # Camera to Lidar
    
    for cam in cams:
        cam_token = first_sample['data'][cam]
        c2e, cs = get_sensor_transform(nusc, cam_token)
        
        # Camera -> Ego -> Lidar
        c2l = e2l @ c2e
        c2l_list.append(c2l.astype(np.float32))
        
        # Intrinsics (3x3 -> 4x4 for padding)
        K = np.eye(4)
        K[:3, :3] = np.array(cs['camera_intrinsic'])
        intrinsics_list.append(K.astype(np.float32))

    # Image Augmentation Matrix (Identity for raw images)
    # The model expects this to know how images were resized/cropped
    # Standard BEVFusion input is usually 1600x900 original -> 704x256 model
    # For simplicity, we pass Identity and let ROS node handle resize
    aug_mat = np.eye(4, dtype=np.float32)
    aug_list = [aug_mat] * 6
    
    # Save Tensors using CUDA-BEVFusion's tensor format (NumPy-based, no PyTorch)
    print("Saving Calibration Tensors...")
    save_tensor(np.stack(c2l_list), os.path.join(OUTPUT_DIR, "camera2lidar.tensor"))
    save_tensor(np.stack(intrinsics_list), os.path.join(OUTPUT_DIR, "camera_intrinsics.tensor"))
    save_tensor(np.stack(aug_list), os.path.join(OUTPUT_DIR, "img_aug_matrix.tensor"))
    
    # Lidar2Image (Projection) = Intrinsic @ Inv(Extrinsic)
    # (This is approximate, the model often calculates this internally or expects pre-calc)
    # For this script, we save a dummy or calculate if strictly needed. 
    # The C++ core uses 'camera2lidar' and 'intrinsics' mostly. 
    # We will save a placeholder for lidar2image to satisfy loader.
    save_tensor(np.zeros((6, 4, 4), dtype=np.float32), os.path.join(OUTPUT_DIR, "lidar2image.tensor"))

    # --- 2. EXTRACT SEQUENCE FRAMES ---
    curr_token = scene['first_sample_token']
    frame_idx = 0
    
    while curr_token != '':
        print(f"Processing Frame {frame_idx}...")
        sample = nusc.get('sample', curr_token)
        
        # Save Images
        for i, cam in enumerate(cams):
            sd = nusc.get('sample_data', sample['data'][cam])
            src_path = os.path.join(NUSC_ROOT, sd['filename'])
            dst_path = os.path.join(OUTPUT_DIR, f"{frame_idx:04d}_{i}_{cam}.jpg")
            
            # Copy/Symlink is faster, but let's write to ensure access
            img = cv2.imread(src_path)
            cv2.imwrite(dst_path, img)

        # Save LiDAR
        lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        lidar_path = os.path.join(NUSC_ROOT, lidar_sd['filename'])
        
        # NuScenes Lidar is (N, 5): x, y, z, intensity, ring_index
        # We need to load binary and save as .npy
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
        np.save(os.path.join(OUTPUT_DIR, f"{frame_idx:04d}_points.npy"), points)

        curr_token = sample['next']
        frame_idx += 1

    print(f"Done! Converted {frame_idx} frames to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()