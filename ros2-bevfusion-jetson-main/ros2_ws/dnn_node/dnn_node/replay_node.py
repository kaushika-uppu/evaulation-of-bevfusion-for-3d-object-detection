import os
import time
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from cv_bridge import CvBridge
import struct

# Points to the sequence folder you created
DATA_ROOT = "/root/ros2_ws/data/bev_sequence"

class ReplayNode(Node):
    def __init__(self):
        super().__init__('replay_node')
        
        # Parameters for pre-resizing optimization
        self.declare_parameter("pre_resize", True)  # Enable pre-resizing at source
        self.declare_parameter("target_width", 1600)  # Target image width
        self.declare_parameter("target_height", 900)  # Target image height
        
        self.pre_resize = self.get_parameter("pre_resize").value
        self.target_w = self.get_parameter("target_width").value
        self.target_h = self.get_parameter("target_height").value
        
        self.bridge = CvBridge()
        self.pubs = {}
        
        # Topics must match bev_node.py exactly
        self.topics = [
            "/cam_front/image_raw", 
            "/cam_front_right/image_raw", 
            "/cam_front_left/image_raw", 
            "/cam_back/image_raw", 
            "/cam_back_left/image_raw", 
            "/cam_back_right/image_raw"
        ]
        
        # Names used in your convert_nuscenes.py script
        self.cam_names = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]

        for topic in self.topics:
            self.pubs[topic] = self.create_publisher(Image, topic, 10)
        
        self.lidar_pub = self.create_publisher(PointCloud2, "/lidar_top/points", 10)
        
        # Parameters
        self.declare_parameter("loop", False)  # Loop continuously or play once
        self.declare_parameter("publish_rate", 5.0)  # Hz (frames per second) - default matches BEVFusion speed
        
        self.loop = self.get_parameter("loop").value
        publish_rate = self.get_parameter("publish_rate").value
        
        # State
        self.frame_idx = 0
        self.finished = False
        
        # check if data exists
        if not os.path.exists(DATA_ROOT):
            self.get_logger().error(f"Data directory not found: {DATA_ROOT}")
            self.get_logger().error("Did you run convert_nuscenes.py?")

        # Log pre-resize configuration
        if self.pre_resize:
            self.get_logger().info(f"Pre-resize ENABLED: Images will be resized to {self.target_w}x{self.target_h} at source")
            self.get_logger().info("This saves ~5-15ms per frame in bev_node by avoiding resize operations")
        else:
            self.get_logger().info("Pre-resize DISABLED: Images will be published at original size")
        
        # Log loop configuration
        if self.loop:
            self.get_logger().info("Loop mode: ENABLED (will continuously loop through dataset)")
        else:
            self.get_logger().info("Loop mode: DISABLED (will play through dataset once)")

        # Create timer based on publish rate
        timer_period = 1.0 / publish_rate
        self.timer = self.create_timer(timer_period, self.publish_frame)
        self.get_logger().info(f"Replay Node Reading from {DATA_ROOT} at {publish_rate} Hz")

    def publish_frame(self):
        # Stop if finished and not looping
        if self.finished:
            return
        
        # Check if current frame exists (using LiDAR file as the check)
        lidar_file = os.path.join(DATA_ROOT, f"{self.frame_idx:04d}_points.npy")
        
        if not os.path.exists(lidar_file):
            if self.loop:
                # Loop mode: restart from beginning
                if self.frame_idx > 0:
                    self.get_logger().info("End of sequence. Looping back to start.")
                self.frame_idx = 0
                # Check again to avoid crash if folder is empty
                lidar_file = os.path.join(DATA_ROOT, f"{self.frame_idx:04d}_points.npy")
                if not os.path.exists(lidar_file):
                    self.get_logger().error("No data files found. Stopping.")
                    self.finished = True
                    return
            else:
                # Play once mode: stop at end
                self.get_logger().info(f"End of sequence reached at frame {self.frame_idx}. Stopping (loop disabled).")
                self.finished = True
                self.timer.cancel()  # Stop the timer
                return

        timestamp = self.get_clock().now().to_msg()
        
        # 1. Load and Publish LiDAR
        try:
            points = np.load(lidar_file)
            pc_msg = self.create_cloud(points, timestamp)
            self.lidar_pub.publish(pc_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to load LiDAR {lidar_file}: {e}")

        # 2. Load and Publish Images
        for i, topic in enumerate(self.topics):
            # Filename format from converter: 0000_0_CAM_FRONT.jpg
            cam_name = self.cam_names[i]
            img_filename = f"{self.frame_idx:04d}_{i}_{cam_name}.jpg"
            img_path = os.path.join(DATA_ROOT, img_filename)
            
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    # OPTIMIZATION: Pre-resize at source to save processing time in bev_node
                    if self.pre_resize:
                        # Check if resize is needed
                        h, w = img.shape[:2]
                        if h != self.target_h or w != self.target_w:
                            # Resize to target dimensions
                            img = cv2.resize(img, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)
                    
                    msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
                    msg.header.stamp = timestamp
                    msg.header.frame_id = "base_link"
                    self.pubs[topic].publish(msg)
            else:
                # Only warn once per missing file to avoid spam
                if self.frame_idx == 0:
                    self.get_logger().warn(f"Missing image: {img_filename}")

        self.get_logger().info(f"Published Frame {self.frame_idx}")
        self.frame_idx += 1
        
        # Check if we've reached the end (for play-once mode)
        next_lidar_file = os.path.join(DATA_ROOT, f"{self.frame_idx:04d}_points.npy")
        if not os.path.exists(next_lidar_file) and not self.loop:
            # Next frame doesn't exist and we're not looping - will stop on next call
            pass

    def create_cloud(self, points, timestamp):
        msg = PointCloud2()
        msg.header.stamp = timestamp
        msg.header.frame_id = "lidar_top"
        
        msg.height = 1
        msg.width = points.shape[0]
        
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        
        msg.is_bigendian = False
        msg.point_step = 20 # 5 floats * 4 bytes
        msg.row_step = msg.point_step * points.shape[0]
        msg.is_dense = True
        
        msg.data = points.astype(np.float32).tobytes()
        return msg

def main(args=None):
    rclpy.init(args=args)
    node = ReplayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()