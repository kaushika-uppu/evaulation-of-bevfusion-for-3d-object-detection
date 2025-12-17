import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import os

# TensorRT & CUDA Imports
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class TRTWrapper:
    """
    Modern TensorRT Wrapper (Compatible with JetPack 6 / TRT 8.6+)
    """
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        # 1. Load Engine
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        self.inputs = []
        self.outputs = []
        self.stream = cuda.Stream()

        # 2. Allocate Memory (Modern API iterates by index, not binding names)
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            
            # Check mode (Input vs Output)
            mode = self.engine.get_tensor_mode(name)
            
            # Get shape & dtype
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size = trt.volume(shape)
            
            # Allocate Host (CPU) and Device (GPU) memory
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            # CRITICAL: Register the memory address with the context
            self.context.set_tensor_address(name, int(device_mem))
            
            tensor_info = {'name': name, 'host': host_mem, 'device': device_mem}
            
            if mode == trt.TensorIOMode.INPUT:
                self.inputs.append(tensor_info)
            else:
                self.outputs.append(tensor_info)

    def infer(self, img):
        # 3. Copy Input (CPU -> GPU)
        np.copyto(self.inputs[0]['host'], img.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # 4. Execute (Use V3 for JetPack 6)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        
        # 5. Copy Output (GPU -> CPU)
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        
        return self.outputs[0]['host']

class ImageClassifier(Node):
    def __init__(self):
        super().__init__('image_classifier')
        
        # Path Config
        package_dir = os.path.dirname(os.path.realpath(__file__))
        engine_path = os.path.join(package_dir, 'models/resnet18.engine')
        labels_path = os.path.join(package_dir, 'models/imagenet_classes.txt')

        self.get_logger().info(f"Loading TensorRT Engine: {engine_path}")
        try:
            self.trt_model = TRTWrapper(engine_path)
            with open(labels_path, "r") as f:
                self.labels = [s.strip() for s in f.readlines()]
            self.get_logger().info("Model Loaded Successfully!")
        except Exception as e:
            self.get_logger().error(f"FATAL Loading Error: {e}")
            return

        # ROS Setup
        self.sub = self.create_subscription(Image, '/image_raw', self.callback, 10)
        self.pub = self.create_publisher(String, '/classification', 10)
        self.bridge = CvBridge()

    def preprocess(self, img):
        # Resize & Normalize for ResNet
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        
        img = img.transpose(2, 0, 1) # HWC -> CHW
        return np.expand_dims(img, axis=0)

    def callback(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Inference Pipeline
            input_tensor = self.preprocess(cv_img)
            output_raw = self.trt_model.infer(input_tensor)
            
            # Post-processing
            idx = np.argmax(output_raw)
            label = self.labels[idx]
            
            self.get_logger().info(f"Detected: {label}")
            self.pub.publish(String(data=label))
            
        except Exception as e:
            self.get_logger().error(f"Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ImageClassifier()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()