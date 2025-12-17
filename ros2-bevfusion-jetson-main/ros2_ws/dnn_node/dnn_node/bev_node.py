import sys
import os
import time
import json
import csv
import traceback
from datetime import datetime
from typing import List
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from vision_msgs.msg import Detection3DArray, Detection3D, ObjectHypothesisWithPose
import message_filters
import importlib.util

# --- PATH SETUP ---
# Find the repo root dynamically
script_dir = os.path.dirname(os.path.realpath(__file__))
# We assume structure: src/dnn_node/dnn_node/bev_node.py -> repo is in ros2_ws/Lidar_AI_Solution
# Adjust logic to find the Lidar_AI_Solution folder
REPO_ROOT = "/root/ros2_ws/Lidar_AI_Solution/CUDA-BEVFusion"

# 1. Add Build Library Path
build_path = os.path.join(REPO_ROOT, "build")
if build_path not in sys.path:
    sys.path.append(build_path)

# 2. Potential helper locations (tensor.py was moved in recent drops)
helper_paths = [
    os.path.join(REPO_ROOT, "tool"),
    os.path.join(REPO_ROOT, "src/common"),
]
for path in helper_paths:
    if path not in sys.path:
        sys.path.append(path)

# Import C++ Library
try:
    import libpybev as pybev
except ImportError as e:
    print(f"FATAL: Could not import libpybev.so from {build_path}.")
    raise e

# Import Tensor Helper (check both tool/ and src/common/)
try:
    tensor_mod = None
    for candidate in helper_paths:
        tensor_file = os.path.join(candidate, "tensor.py")
        if os.path.exists(tensor_file):
            tensor_spec = importlib.util.spec_from_file_location("tensor_helper", tensor_file)
            tensor_mod = importlib.util.module_from_spec(tensor_spec)
            tensor_spec.loader.exec_module(tensor_mod)
            load_tensor = tensor_mod.load
            break
    if tensor_mod is None:
        raise FileNotFoundError("tensor.py not found under tool/ or src/common/")
except Exception as e:
    print(f"FATAL: Could not load tensor.py: {e}")
    raise e

class BEVFusionNode(Node):
    CLASS_NAMES = [
        "car", "truck", "construction_vehicle", "bus", "trailer",
        "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone"
    ]

    def __init__(self):
        super().__init__('bevfusion_node')
        
        # Parameters
        self.declare_parameter("confidence_threshold", 0.01)  # Lowered temporarily to debug low scores
        self.conf_threshold = self.get_parameter("confidence_threshold").value
        
        # Logging parameters
        self.declare_parameter("log_dir", "/root/ros2_ws/logs/bevfusion")
        self.declare_parameter("save_logs", True)
        self.declare_parameter("save_metrics", True)
        self.declare_parameter("save_detections", True)  # Changed default to True
        
        log_dir = self.get_parameter("log_dir").value
        self.save_logs = self.get_parameter("save_logs").value
        self.save_metrics = self.get_parameter("save_metrics").value
        self.save_detections = self.get_parameter("save_detections").value
        
        # Setup logging directory
        if self.save_logs or self.save_metrics or self.save_detections:
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_dir = os.path.join(log_dir, timestamp)
            os.makedirs(self.log_dir, exist_ok=True)
            
            # Set permissions to allow deletion (make world-writable if possible)
            try:
                os.chmod(self.log_dir, 0o777)
                # Also set permissions on parent if we can
                os.chmod(log_dir, 0o777)
            except (OSError, PermissionError):
                pass  # Ignore permission errors - may not have permission to change
            
            # Setup log files
            if self.save_logs:
                self.log_file = open(os.path.join(self.log_dir, "node.log"), "w")
                self.get_logger().info(f"Logging to: {self.log_file.name}")
            
            # Setup metrics CSV
            if self.save_metrics:
                self.metrics_file = os.path.join(self.log_dir, "metrics.csv")
                with open(self.metrics_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'timestamp', 'frame_id', 'num_points', 'num_detections',
                        'prep_time_ms', 'inference_time_ms', 'publish_time_ms', 'total_time_ms', 'fps'
                    ])
                self.get_logger().info(f"Metrics logging to: {self.metrics_file}")
            
            # Setup detections JSON
            if self.save_detections:
                self.detections_file = os.path.join(self.log_dir, "detections.jsonl")
                self.get_logger().info(f"Detections logging to: {self.detections_file}")
        else:
            self.log_dir = None
            self.log_file = None
            self.metrics_file = None
            self.detections_file = None
        
        # Statistics tracking
        self.frame_count = 0
        self.total_prep_time = 0.0
        self.total_inf_time = 0.0
        self.total_pub_time = 0.0
        
        # Initialize image buffers for zero-copy processing
        self._init_buffers()
        
        # Model Config
        model_variant = "resnet50"  # Match the model you built
        model_root = os.path.join(REPO_ROOT, "model", model_variant, "build")
        # Use converted nuScenes data (or example-data for quick start)
        # Option 1: Your converted data (recommended if you have nuScenes)
        converted_data_dir = "/home/student/ros2/ros2_ws/data/bev_sequence"
        # Option 2: Example data (fallback)
        example_data_dir = os.path.join(REPO_ROOT, "example-data")
        # Use converted data if it exists, otherwise fall back to example-data
        if os.path.exists(converted_data_dir) and os.path.exists(os.path.join(converted_data_dir, "camera2lidar.tensor")):
            calib_dir = converted_data_dir
            self.get_logger().info(f"Using converted calibration data: {calib_dir}")
        else:
            calib_dir = example_data_dir
            self.get_logger().info(f"Using example calibration data: {calib_dir}")
        
        self.get_logger().info(f"Loading Model from: {model_root}")
        
        # Initialize Core (FP16)
        self.core = pybev.load_bevfusion(
            os.path.join(model_root, "camera.backbone.plan"),
            os.path.join(model_root, "camera.vtransform.plan"),
            os.path.join(REPO_ROOT, "model", model_variant, "lidar.backbone.xyz.onnx"),
            os.path.join(model_root, "fuser.plan"),
            os.path.join(model_root, "head.bbox.plan"),
            "fp16"
        )
        
        if self.core is None:
             raise RuntimeError("Core Init Failed")
        
        # Print model architecture info
        self.core.print()
        
        # Enable built-in C++ timer for detailed per-stage timing (requires rebuilt libpybev)
        try:
            self.core.set_timer(True)
            self.get_logger().info("Built-in C++ timer enabled - will show per-stage timing")
        except AttributeError:
            self.get_logger().warn("set_timer() not available - rebuild libpybev to enable C++ timing. Using Python-level timing only.")
        
        self._load_calibration(calib_dir)

        # Setup ROS
        self.bridge = CvBridge()
        self.subs = []
        topics = [
            '/cam_front/image_raw', '/cam_front_right/image_raw', '/cam_front_left/image_raw',
            '/cam_back/image_raw', '/cam_back_left/image_raw', '/cam_back_right/image_raw'
        ]
        
        qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        
        for t in topics:
            self.subs.append(message_filters.Subscriber(self, Image, t, qos_profile=qos))
        
        self.lidar_sub = message_filters.Subscriber(self, PointCloud2, '/lidar_top/points', qos_profile=qos)

        # Sync Policy
        self.ts = message_filters.ApproximateTimeSynchronizer(self.subs + [self.lidar_sub], 10, 0.1)
        self.ts.registerCallback(self._sync_callback)

        self.pub = self.create_publisher(Detection3DArray, '/bevfusion/detections', 10)
        self.get_logger().info("Node Ready! Waiting for synced data...")

    def _load_calibration(self, directory):
        self.get_logger().info(f"Loading calibration from {directory}...")
        try:
            cam2lidar = load_tensor(os.path.join(directory, "camera2lidar.tensor"))
            intrinsics = load_tensor(os.path.join(directory, "camera_intrinsics.tensor"))
            lidar2img = load_tensor(os.path.join(directory, "lidar2image.tensor"))
            img_aug = load_tensor(os.path.join(directory, "img_aug_matrix.tensor"))
            
            # Debug: Log calibration shapes
            self.get_logger().info(f"Calibration loaded - cam2lidar: {cam2lidar.shape}, intrinsics: {intrinsics.shape}, "
                                 f"lidar2img: {lidar2img.shape}, img_aug: {img_aug.shape}")
            
            self.core.update(cam2lidar, intrinsics, lidar2img, img_aug)
            self.get_logger().info("Calibration updated successfully")
        except Exception as e:
            self.get_logger().error(f"Calibration Error: {e}\n{traceback.format_exc()}")
            raise

    def _init_buffers(self):
        # Call this in __init__
        # Pre-allocate a single contiguous block for 6 images
        # Shape: (1, 6, 900, 1600, 3) | Type: uint8
        self.image_buffer = np.zeros((1, 6, 900, 1600, 3), dtype=np.uint8)
        
        # Create views for each camera
        # These views point to the same memory as self.image_buffer
        self.cam_views = [self.image_buffer[0, i, ...] for i in range(6)]

    def _prepare_images(self, images: List[Image]) -> np.ndarray:
        target_w, target_h = 1600, 900 

        for i, msg in enumerate(images):
            # We still have to allocate for the decode (cv_bridge limitation)
            # But we avoid the allocation for the list and the final stack
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            
            # OPTIMIZATION: If images are pre-resized at source (replay_node), skip resize
            # This saves ~5-15ms per frame when using pre-resized images
            if cv_image.shape[0] != target_h or cv_image.shape[1] != target_w:
                # Resize needed (images not pre-resized at source)
                # cv2.resize can write to a pre-allocated destination (dst)
                # BUT only if src and dst are different.
                # Since we are resizing AND coloring, we do it in steps.
                temp = cv2.resize(cv_image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                cv2.cvtColor(temp, cv2.COLOR_BGR2RGB, dst=self.cam_views[i])
            else:
                # Images already correct size (pre-resized at source) - just convert color
                # Direct color conversion into the buffer (saves resize time!)
                cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB, dst=self.cam_views[i])

        # Return the pre-allocated buffer (Zero copy!)
        return self.image_buffer

    def _prepare_points(self, cloud: PointCloud2) -> np.ndarray:
        # Read x,y,z,intensity
        # We use a manual buffer read for speed and to ensure float32 first
        raw_data = np.frombuffer(cloud.data, dtype=np.float32)
        point_step = cloud.point_step // 4
        num_points = cloud.width * cloud.height
        points = raw_data.reshape(num_points, point_step)
        
        # Extract first 4 columns
        xyz_i = points[:, :4]
        
        # Add padding for 5th column if needed by C++
        padding = np.zeros((num_points, 1), dtype=np.float32)
        final_points = np.hstack((xyz_i, padding))
        
        # CRITICAL: Cast to Float16 for the FP16 model
        return np.ascontiguousarray(final_points.astype(np.float16))

    def _sync_callback(self, *msgs):
        # msgs[0-5] = Images, msgs[6] = Lidar
        img_msgs = msgs[:6]
        lidar_msg = msgs[6]
        
        try:
            # 1. Prepare Inputs (with timing)
            prep_start = time.perf_counter()
            images = self._prepare_images(img_msgs)
            points = self._prepare_points(lidar_msg)
            prep_time = (time.perf_counter() - prep_start) * 1000  # Convert to ms
            
            if points.size == 0:
                self.get_logger().warn("Empty point cloud, skipping")
                return

            # Debug: Log input shapes
            self.get_logger().debug(f"Images shape: {images.shape}, dtype: {images.dtype}, range: [{images.min()}, {images.max()}]")
            self.get_logger().debug(f"Points shape: {points.shape}, dtype: {points.dtype}, num_points: {points.shape[0]}")

            # 2. Inference (with timing)
            # Note: For detailed per-stage C++ timing, rebuild libpybev with set_timer() exposed
            inf_start = time.perf_counter()
            detections = self.core.forward(images, points, True, False)
            inf_time = (time.perf_counter() - inf_start) * 1000  # Convert to ms
            
            # Debug: Log raw detections
            if len(detections) > 0:
                # Check for NaN/inf in detections
                nan_count = np.sum(~np.isfinite(detections))
                valid_scores = detections[:, 10] if detections.shape[1] > 10 else []
                if len(valid_scores) > 0:
                    finite_scores = valid_scores[np.isfinite(valid_scores)]
                    if len(finite_scores) > 0:
                        # Score distribution analysis
                        above_01 = np.sum(finite_scores >= 0.01)
                        above_05 = np.sum(finite_scores >= 0.05)
                        above_10 = np.sum(finite_scores >= 0.10)
                        above_25 = np.sum(finite_scores >= 0.25)
                        score_stats = f"score range: [{np.min(finite_scores):.3f}, {np.max(finite_scores):.3f}], valid: {len(finite_scores)}/{len(valid_scores)}, above thresholds: 0.01={above_01}, 0.05={above_05}, 0.10={above_10}, 0.25={above_25}"
                    else:
                        score_stats = "all NaN"
                else:
                    score_stats = "no scores"
                self.get_logger().info(f"Raw detections: {len(detections)} total, {nan_count} NaN/inf values, {score_stats}")
                if len(detections) > 0 and detections.shape[1] >= 11:
                    # Log top 5 detections by score
                    sorted_indices = np.argsort(detections[:, 10])[::-1] if len(valid_scores) > 0 else range(min(5, len(detections)))
                    for idx, i in enumerate(sorted_indices[:5]):
                        det = detections[i]
                        self.get_logger().info(f"  Top[{idx}]: xyz=({det[0]:.2f},{det[1]:.2f},{det[2]:.2f}), size=({det[3]:.2f},{det[4]:.2f},{det[5]:.2f}), score={det[10]:.3f}, class={int(det[9])}")
            else:
                self.get_logger().info("No detections returned from model")
            
            # 3. Publish (with timing)
            pub_start = time.perf_counter()
            self._publish_detections(detections, lidar_msg.header)
            pub_time = (time.perf_counter() - pub_start) * 1000  # Convert to ms
            
            # Log Python-level timing summary
            total_time = prep_time + inf_time + pub_time
            fps = 1000 / total_time if total_time > 0 else 0
            
            log_msg = (
                f"[Python Timing] Prep: {prep_time:.1f}ms | "
                f"Inference: {inf_time:.1f}ms | "
                f"Publish: {pub_time:.1f}ms | "
                f"Total: {total_time:.1f}ms ({fps:.1f} FPS)"
            )
            self.get_logger().info(log_msg)
            
            # Write to log file if enabled
            if self.log_file:
                self.log_file.write(f"{datetime.now().isoformat()} [INFO] {log_msg}\n")
                self.log_file.flush()
            
            # Save metrics to CSV
            if self.save_metrics and self.metrics_file:
                self.frame_count += 1
                self.total_prep_time += prep_time
                self.total_inf_time += inf_time
                self.total_pub_time += pub_time
                
                num_detections = len(detections) if len(detections) > 0 else 0
                num_points = points.shape[0] if points.size > 0 else 0
                frame_id = lidar_msg.header.frame_id if hasattr(lidar_msg.header, 'frame_id') else str(self.frame_count)
                
                with open(self.metrics_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        datetime.now().isoformat(),
                        frame_id,
                        num_points,
                        num_detections,
                        f"{prep_time:.2f}",
                        f"{inf_time:.2f}",
                        f"{pub_time:.2f}",
                        f"{total_time:.2f}",
                        f"{fps:.2f}"
                    ])
            
            # Save detections to JSONL
            if self.save_detections and self.detections_file:
                detections_data = {
                    'timestamp': datetime.now().isoformat(),
                    'frame_id': lidar_msg.header.frame_id if hasattr(lidar_msg.header, 'frame_id') else str(self.frame_count),
                    'num_points': points.shape[0] if points.size > 0 else 0,
                    'detections': []
                }
                
                # Process detections even if empty (to track frames with no detections)
                if len(detections) > 0:
                    for det in detections:
                        if isinstance(det, np.ndarray) and len(det) >= 11:
                            score = float(det[10])
                            if np.isfinite(score) and score >= self.conf_threshold:
                                detections_data['detections'].append({
                                    'position': [float(det[0]), float(det[1]), float(det[2])],
                                    'size': [float(det[3]), float(det[4]), float(det[5])],
                                    'rotation': float(det[6]),
                                    'velocity': [float(det[7]), float(det[8])],
                                    'class_id': int(det[9]),
                                    'class_name': self.CLASS_NAMES[int(det[9])] if 0 <= int(det[9]) < len(self.CLASS_NAMES) else 'unknown',
                                    'score': score
                                })
                
                # Always write, even if no detections (to track all frames)
                try:
                    with open(self.detections_file, 'a') as f:
                        f.write(json.dumps(detections_data) + '\n')
                except Exception as e:
                    self.get_logger().error(f"Failed to write detections: {e}\n{traceback.format_exc()}")
            
        except Exception as e:
            self.get_logger().error(f"Inference Fail: {e}\n{traceback.format_exc()}")

    def _publish_detections(self, detections, header):
        out_msg = Detection3DArray()
        out_msg.header = header
        
        if len(detections) == 0:
            self.pub.publish(out_msg)
            return
        
        valid_count = 0
        for det in detections:
            # Handle NaN/inf values
            if not isinstance(det, np.ndarray) or len(det) < 11:
                continue
                
            score = float(det[10])
            
            # Skip invalid scores (NaN, inf, or below threshold)
            if not np.isfinite(score) or score < self.conf_threshold:
                continue
            
            # Check for NaN in position/size
            if not all(np.isfinite([det[0], det[1], det[2], det[3], det[4], det[5]])):
                continue
            
            ros_det = Detection3D()
            ros_det.header = header
            
            # BBox: x, y, z, w, l, h, rot
            ros_det.bbox.center.position.x = float(det[0])
            ros_det.bbox.center.position.y = float(det[1])
            ros_det.bbox.center.position.z = float(det[2])
            ros_det.bbox.size.x = float(det[4])  # width
            ros_det.bbox.size.y = float(det[3])  # length
            ros_det.bbox.size.z = float(det[5])  # height
            
            yaw = float(det[6])
            if np.isfinite(yaw):
                ros_det.bbox.center.orientation.z = np.sin(yaw / 2.0)
                ros_det.bbox.center.orientation.w = np.cos(yaw / 2.0)
            else:
                ros_det.bbox.center.orientation.w = 1.0  # Default orientation
            
            # Class
            class_id = int(det[9]) if np.isfinite(det[9]) else 0
            if 0 <= class_id < len(self.CLASS_NAMES):
                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = self.CLASS_NAMES[class_id]
                hyp.hypothesis.score = score
                ros_det.results.append(hyp)
            else:
                # Unknown class, skip
                continue
                
            out_msg.detections.append(ros_det)
            valid_count += 1
            
        self.get_logger().info(f"Published {valid_count}/{len(detections)} valid detections (threshold: {self.conf_threshold})")
        self.pub.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = BEVFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        # Print summary statistics BEFORE destroying node
        summary = None
        if node.frame_count > 0:
            avg_prep = node.total_prep_time / node.frame_count
            avg_inf = node.total_inf_time / node.frame_count
            avg_pub = node.total_pub_time / node.frame_count
            avg_total = avg_prep + avg_inf + avg_pub
            avg_fps = 1000 / avg_total if avg_total > 0 else 0
            
            summary = (
                f"\n{'='*60}\n"
                f"SUMMARY STATISTICS ({node.frame_count} frames)\n"
                f"{'='*60}\n"
                f"Average Prep Time:    {avg_prep:.2f} ms\n"
                f"Average Inference:    {avg_inf:.2f} ms\n"
                f"Average Publish:      {avg_pub:.2f} ms\n"
                f"Average Total:        {avg_total:.2f} ms\n"
                f"Average FPS:          {avg_fps:.2f}\n"
                f"{'='*60}\n"
            )
            # Log summary while node is still valid
            try:
                node.get_logger().info(summary)
            except:
                pass  # Ignore if logging fails
            
            # Write to log file
            if node.log_file:
                try:
                    node.log_file.write(summary)
                    node.log_file.close()
                except:
                    pass
            
            # Save summary file
            if node.save_metrics and node.log_dir:
                try:
                    summary_file = os.path.join(node.log_dir, "summary.txt")
                    with open(summary_file, 'w') as f:
                        f.write(summary)
                    # Try to log, but don't fail if node is already destroyed
                    try:
                        node.get_logger().info(f"Summary saved to: {summary_file}")
                    except:
                        print(f"Summary saved to: {summary_file}")
                except Exception as e:
                    print(f"Failed to save summary: {e}")
        
        # Destroy node first
        try:
            node.destroy_node()
        except:
            pass
        
        # Then shutdown (only if not already shut down)
        try:
            rclpy.shutdown()
        except:
            pass  # Ignore if already shut down

if __name__ == '__main__':
    main()