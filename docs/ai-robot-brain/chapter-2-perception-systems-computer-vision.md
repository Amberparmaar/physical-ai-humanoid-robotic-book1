---
sidebar_position: 2
title: Perception Systems and Computer Vision
---

# Perception Systems and Computer Vision

In this chapter, we'll explore the fundamental role of perception systems in AI robot brains, with a focus on computer vision. Perception is the foundation of intelligent behavior, enabling robots to understand and interact with their environment through visual, auditory, and other sensory inputs.

## Understanding Perception in Robotics

Perception in robotics involves processing sensory data to extract meaningful information about the environment. This includes:

### 1. Environmental Understanding
- Object detection and recognition
- Scene segmentation and understanding
- Spatial relationships and layout
- Dynamic element tracking

### 2. Self-Localization
- Position and orientation estimation
- Map building and maintenance
- Path planning and navigation
- Obstacle detection and avoidance

### 3. Interaction Preparation
- Grasp point identification
- Surface analysis for manipulation
- Human detection and pose estimation
- Social context understanding

## Computer Vision Fundamentals

### Image Formation and Processing

```python
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

class ImageProcessor:
    def __init__(self):
        # Standard image preprocessing transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def preprocess_image(self, image):
        """Preprocess image for deep learning models"""
        # Convert BGR to RGB (OpenCV uses BGR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to standard size
        image_resized = cv2.resize(image_rgb, (224, 224))

        # Apply normalization
        image_tensor = self.transform(image_resized)

        return image_tensor.unsqueeze(0)  # Add batch dimension

    def detect_edges(self, image):
        """Detect edges in image using Canny edge detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return edges

    def extract_features(self, image):
        """Extract SIFT features from image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        return keypoints, descriptors
```

### Object Detection and Recognition

```python
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms as T

class ObjectDetector:
    def __init__(self, model_path=None):
        # Load pre-trained object detection model
        if model_path:
            self.model = torch.load(model_path)
        else:
            self.model = fasterrcnn_resnet50_fpn(pretrained=True)

        self.model.eval()
        self.transforms = T.Compose([
            T.ToTensor(),
        ])

        # COCO dataset class names
        self.coco_names = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
            'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
            'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
            'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
            'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

    def detect_objects(self, image):
        """Detect objects in image using deep learning model"""
        # Preprocess image
        image_tensor = self.transforms(image).unsqueeze(0)

        # Run inference
        with torch.no_grad():
            predictions = self.model(image_tensor)

        # Extract results
        boxes = predictions[0]['boxes'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()

        # Filter by confidence threshold
        confidence_threshold = 0.5
        valid_indices = scores > confidence_threshold

        results = []
        for i in valid_indices.nonzero()[0]:
            result = {
                'bbox': boxes[i],
                'label': self.coco_names[labels[i]],
                'confidence': scores[i]
            }
            results.append(result)

        return results

    def visualize_detections(self, image, detections):
        """Visualize object detections on image"""
        vis_image = image.copy()

        for detection in detections:
            bbox = detection['bbox']
            label = detection['label']
            confidence = detection['confidence']

            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            text = f"{label}: {confidence:.2f}"
            cv2.putText(vis_image, text, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return vis_image
```

### Semantic Segmentation

```python
import torch
import torchvision.transforms as T
from torchvision.models.segmentation import fcn_resnet50

class SemanticSegmenter:
    def __init__(self):
        # Load pre-trained segmentation model
        self.model = fcn_resnet50(pretrained=True)
        self.model.eval()

        # COCO dataset class names for segmentation
        self.coco_classes = [
            'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
            'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor'
        ]

        # Color map for visualization
        self.color_map = np.random.randint(0, 255, size=(len(self.coco_classes), 3))

    def segment_image(self, image):
        """Perform semantic segmentation on image"""
        # Preprocess image
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
        ])

        image_tensor = transform(image).unsqueeze(0)

        # Run inference
        with torch.no_grad():
            output = self.model(image_tensor)['out'][0]
            predicted = torch.argmax(output, dim=0).cpu().numpy()

        return predicted

    def visualize_segmentation(self, image, segmentation):
        """Visualize segmentation results"""
        # Create color overlay
        height, width = segmentation.shape
        color_seg = np.zeros((height, width, 3), dtype=np.uint8)

        for label in np.unique(segmentation):
            if label < len(self.color_map):
                color_seg[segmentation == label] = self.color_map[label]

        # Blend with original image
        alpha = 0.7
        blended = cv2.addWeighted(image, alpha, color_seg, 1 - alpha, 0)

        return blended
```

## NVIDIA Isaac Perception Pipeline

### Isaac ROS Perception Nodes

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose
import numpy as np

class IsaacPerceptionNode(Node):
    def __init__(self):
        super().__init__('isaac_perception_node')

        # Initialize CV bridge
        self.cv_bridge = CvBridge()

        # Initialize perception components
        self.object_detector = ObjectDetector()
        self.segmenter = SemanticSegmenter()

        # Create subscribers and publishers
        self.image_sub = self.create_subscription(
            Image, '/camera/rgb/image_rect_color', self.image_callback, 10)

        self.detection_pub = self.create_publisher(
            Detection2DArray, '/isaac/detections', 10)

        self.segmentation_pub = self.create_publisher(
            Image, '/isaac/segmentation', 10)

        self.get_logger().info('Isaac Perception Node initialized')

    def image_callback(self, msg):
        """Process incoming camera images"""
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")

            # Perform object detection
            detections = self.object_detector.detect_objects(cv_image)

            # Perform semantic segmentation
            segmentation = self.segmenter.segment_image(cv_image)

            # Publish results
            self.publish_detections(detections, msg.header)
            self.publish_segmentation(segmentation, msg.header)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def publish_detections(self, detections, header):
        """Publish object detections"""
        detection_array = Detection2DArray()
        detection_array.header = header

        for detection in detections:
            vision_detection = Detection2D()
            vision_detection.header = header

            # Set bounding box
            bbox = detection['bbox']
            vision_detection.bbox.size_x = bbox[2] - bbox[0]
            vision_detection.bbox.size_y = bbox[3] - bbox[1]
            vision_detection.bbox.center.x = (bbox[0] + bbox[2]) / 2
            vision_detection.bbox.center.y = (bbox[1] + bbox[3]) / 2

            # Set class and confidence
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = detection['label']
            hypothesis.hypothesis.score = detection['confidence']
            vision_detection.results.append(hypothesis)

            detection_array.detections.append(vision_detection)

        self.detection_pub.publish(detection_array)

    def publish_segmentation(self, segmentation, header):
        """Publish segmentation results"""
        # Convert segmentation to image message
        seg_image = self.cv_bridge.cv2_to_imgmsg(
            segmentation.astype(np.uint8), "mono8")
        seg_image.header = header
        self.segmentation_pub.publish(seg_image)
```

## 3D Perception and Depth Processing

### Depth Image Processing

```python
import numpy as np
import open3d as o3d
from sensor_msgs.msg import PointCloud2, Image
import sensor_msgs.point_cloud2 as pc2

class DepthProcessor:
    def __init__(self):
        self.camera_intrinsics = None  # To be set from camera info

    def depth_to_pointcloud(self, depth_image, camera_info):
        """Convert depth image to point cloud"""
        # Extract camera parameters
        fx = camera_info.k[0]  # Focal length x
        fy = camera_info.k[4]  # Focal length y
        cx = camera_info.k[2]  # Principal point x
        cy = camera_info.k[5]  # Principal point y

        height, width = depth_image.shape
        points = []

        for v in range(height):
            for u in range(width):
                z = depth_image[v, u]
                if z > 0:  # Valid depth
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy
                    points.append([x, y, z])

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        return pcd

    def segment_objects(self, pointcloud, cluster_tolerance=0.02, min_cluster_size=100):
        """Segment objects in point cloud using Euclidean clustering"""
        # Downsample point cloud
        downsampled = pointcloud.voxel_down_sample(voxel_size=0.01)

        # Estimate normals
        downsampled.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.02, max_nn=30))

        # Perform clustering
        labels = np.array(downsampled.cluster_dbscan(
            eps=cluster_tolerance, min_points=min_cluster_size, print_progress=False))

        # Group points by cluster
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(np.asarray(downsampled.points)[i])

        return clusters

    def extract_planes(self, pointcloud, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
        """Extract planar surfaces (like tables, floors) from point cloud"""
        plane_model, inliers = pointcloud.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations)

        # Extract inlier points (plane) and outlier points (objects)
        plane_cloud = pointcloud.select_by_index(inliers)
        objects_cloud = pointcloud.select_by_index(inliers, invert=True)

        return plane_model, plane_cloud, objects_cloud
```

## Feature Extraction and Matching

### Advanced Feature Detection

```python
class FeatureExtractor:
    def __init__(self):
        # Initialize multiple feature detectors
        self.sift = cv2.SIFT_create()
        self.orb = cv2.ORB_create()
        self.akaze = cv2.AKAZE_create()

    def extract_sift_features(self, image):
        """Extract SIFT features from image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        return keypoints, descriptors

    def extract_orb_features(self, image):
        """Extract ORB features from image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        return keypoints, descriptors

    def extract_akaze_features(self, image):
        """Extract AKAZE features from image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.akaze.detectAndCompute(gray, None)
        return keypoints, descriptors

    def match_features(self, desc1, desc2, matcher_type='bf'):
        """Match features between two images"""
        if matcher_type == 'bf':
            # Brute force matcher
            matcher = cv2.BFMatcher()
            matches = matcher.knnMatch(desc1, desc2, k=2)
        elif matcher_type == 'flann':
            # FLANN matcher (faster for large datasets)
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
            matches = matcher.knnMatch(desc1, desc2, k=2)

        # Apply Lowe's ratio test to filter good matches
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

        return good_matches

    def find_homography(self, kp1, kp2, matches, threshold=5.0):
        """Find homography transformation between matched keypoints"""
        if len(matches) >= 4:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            homography, mask = cv2.findHomography(
                src_pts, dst_pts, cv2.RANSAC, threshold, mask=None)

            return homography, mask
        else:
            return None, None
```

## Multi-Modal Perception Fusion

### Sensor Fusion Framework

```python
import numpy as np
from scipy.spatial.transform import Rotation as R

class SensorFusion:
    def __init__(self):
        self.camera_intrinsics = None
        self.lidar_to_camera_extrinsics = None
        self.imu_bias = np.zeros(6)  # 3 for accelerometer, 3 for gyroscope

    def camera_lidar_fusion(self, image, pointcloud, camera_matrix, transform):
        """Fuse camera and LIDAR data"""
        # Project 3D points to 2D image coordinates
        points_3d = np.asarray(pointcloud.points)

        # Transform points from LIDAR frame to camera frame
        points_3d_cam = self.transform_points(points_3d, transform)

        # Project 3D points to 2D image plane
        points_2d, depth = self.project_3d_to_2d(points_3d_cam, camera_matrix)

        # Filter points within image bounds
        valid_mask = (
            (points_2d[:, 0] >= 0) & (points_2d[:, 0] < image.shape[1]) &
            (points_2d[:, 1] >= 0) & (points_2d[:, 1] < image.shape[0]) &
            (depth > 0)
        )

        valid_points_2d = points_2d[valid_mask]
        valid_points_3d = points_3d[valid_mask]
        valid_depth = depth[valid_mask]

        # Create colorized point cloud
        colorized_points = []
        for i, pt_2d in enumerate(valid_points_2d):
            x, y = int(pt_2d[0]), int(pt_2d[1])
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                color = image[y, x]
                colorized_points.append({
                    'point': valid_points_3d[i],
                    'color': color,
                    'depth': valid_depth[i]
                })

        return colorized_points

    def transform_points(self, points, transform_matrix):
        """Transform points using 4x4 transformation matrix"""
        # Convert to homogeneous coordinates
        points_homo = np.hstack([points, np.ones((points.shape[0], 1))])

        # Apply transformation
        transformed_homo = transform_matrix @ points_homo.T

        # Convert back to 3D
        transformed = transformed_homo[:3, :].T

        return transformed

    def project_3d_to_2d(self, points_3d, camera_matrix):
        """Project 3D points to 2D image coordinates"""
        # Camera matrix: [fx, 0, cx; 0, fy, cy; 0, 0, 1]
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

        # Project points
        x = points_3d[:, 0] / points_3d[:, 2]
        y = points_3d[:, 1] / points_3d[:, 2]

        # Apply intrinsic parameters
        u = fx * x + cx
        v = fy * y + cy

        points_2d = np.column_stack([u, v])
        depth = points_3d[:, 2]

        return points_2d, depth

    def imu_camera_fusion(self, imu_data, camera_pose, dt=0.01):
        """Fuse IMU data with camera pose estimation"""
        # Integrate IMU measurements
        angular_velocity = imu_data['angular_velocity']
        linear_acceleration = imu_data['linear_acceleration']

        # Update orientation using gyroscope
        rotation = R.from_rotvec(angular_velocity * dt)
        new_orientation = rotation * R.from_quat(camera_pose.orientation)

        # Update position using accelerometer (double integration)
        velocity = linear_acceleration * dt
        position = 0.5 * linear_acceleration * dt**2

        # Combine with camera measurements using Kalman filter
        # (Simplified - in practice, use proper Kalman filter implementation)
        fused_pose = self.kalman_filter_update(
            camera_pose, position, new_orientation, dt)

        return fused_pose

    def kalman_filter_update(self, camera_pose, imu_prediction, orientation, dt):
        """Update pose estimate using Kalman filter"""
        # Simplified Kalman filter update
        # In practice, implement full Kalman filter equations

        # Process noise covariance
        Q = np.eye(6) * 0.1  # Process noise

        # Measurement noise covariance
        R = np.eye(6) * 1.0  # Measurement noise

        # State transition matrix (simplified)
        F = np.eye(6)  # Identity for static case

        # Measurement matrix
        H = np.eye(6)  # Direct measurement

        # Predict step would go here in full implementation
        # Update step would go here in full implementation

        # For this example, return a simple weighted average
        alpha = 0.7  # Weight for camera measurement
        beta = 0.3   # Weight for IMU prediction

        # Combine estimates
        fused_position = alpha * camera_pose.position + beta * imu_prediction
        fused_orientation = orientation.as_quat()  # Use IMU orientation

        return {
            'position': fused_position,
            'orientation': fused_orientation
        }
```

## Real-time Perception Optimization

### GPU-Accelerated Perception

```python
import torch
import torchvision.transforms as T
import time

class GPUPerceptionPipeline:
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')

        # Load models to GPU
        self.detection_model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.segmentation_model = fcn_resnet50(pretrained=True)

        self.detection_model.to(self.device)
        self.segmentation_model.to(self.device)

        self.detection_model.eval()
        self.segmentation_model.eval()

        # Create TensorRT optimization if available
        if self.use_gpu:
            try:
                import torch_tensorrt
                self.trt_detection_model = torch_tensorrt.compile(
                    self.detection_model,
                    inputs=[torch.randn(1, 3, 480, 640).to(self.device)],
                    enabled_precisions={torch.float}
                )
                self.trt_segmentation_model = torch_tensorrt.compile(
                    self.segmentation_model,
                    inputs=[torch.randn(1, 3, 480, 640).to(self.device)],
                    enabled_precisions={torch.float}
                )
            except ImportError:
                print("TensorRT not available, using standard PyTorch models")
                self.trt_detection_model = self.detection_model
                self.trt_segmentation_model = self.segmentation_model

    def process_frame_batch(self, frames):
        """Process a batch of frames efficiently"""
        # Stack frames into batch
        batch_tensor = torch.stack([self.preprocess_frame(frame) for frame in frames])
        batch_tensor = batch_tensor.to(self.device)

        # Run inference on batch
        with torch.no_grad():
            # Object detection
            detection_results = self.trt_detection_model(batch_tensor)

            # Semantic segmentation
            segmentation_results = self.trt_segmentation_model(batch_tensor)

        return detection_results, segmentation_results

    def preprocess_frame(self, frame):
        """Preprocess a single frame for model input"""
        # Resize frame
        resized = cv2.resize(frame, (640, 480))

        # Convert BGR to RGB and normalize
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = T.ToTensor()(rgb)

        # Normalize using ImageNet statistics
        normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        normalized = normalize(tensor)

        return normalized

    def get_performance_stats(self):
        """Get performance statistics for optimization"""
        # Monitor GPU utilization
        if self.use_gpu:
            gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            gpu_util = torch.cuda.utilization()  # Percentage
        else:
            gpu_memory = 0
            gpu_util = 0

        return {
            'gpu_memory_gb': gpu_memory,
            'gpu_utilization': gpu_util,
            'device': str(self.device)
        }
```

## Perception Quality Assessment

### Quality Metrics and Validation

```python
class PerceptionQualityAssessor:
    def __init__(self):
        self.metrics_history = []

    def assess_detection_quality(self, detections, ground_truth):
        """Assess quality of object detections"""
        # Calculate precision, recall, and mAP
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        # Match detections to ground truth
        matched_gt = set()
        for detection in detections:
            best_match = None
            best_iou = 0

            for gt in ground_truth:
                iou = self.calculate_iou(detection['bbox'], gt['bbox'])
                if iou > best_iou and iou > 0.5:  # IoU threshold
                    best_iou = iou
                    best_match = gt

            if best_match and best_match['id'] not in matched_gt:
                true_positives += 1
                matched_gt.add(best_match['id'])
            else:
                false_positives += 1

        false_negatives = len(ground_truth) - len(matched_gt)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two bounding boxes"""
        # Calculate intersection
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])

        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0

        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)

        # Calculate union
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = area1 + area2 - inter_area

        return inter_area / union_area

    def assess_segmentation_quality(self, segmentation, ground_truth):
        """Assess quality of semantic segmentation"""
        # Calculate pixel accuracy
        total_pixels = segmentation.size
        correct_pixels = np.sum(segmentation == ground_truth)
        pixel_accuracy = correct_pixels / total_pixels

        # Calculate mean IoU
        iou_scores = []
        unique_labels = np.unique(np.concatenate([segmentation.flatten(), ground_truth.flatten()]))

        for label in unique_labels:
            pred_mask = segmentation == label
            gt_mask = ground_truth == label

            intersection = np.sum(pred_mask & gt_mask)
            union = np.sum(pred_mask | gt_mask)

            if union > 0:
                iou_scores.append(intersection / union)

        mean_iou = np.mean(iou_scores) if iou_scores else 0.0

        return {
            'pixel_accuracy': pixel_accuracy,
            'mean_iou': mean_iou,
            'iou_per_class': dict(zip(unique_labels, iou_scores))
        }

    def detect_perception_anomalies(self, current_results, historical_data):
        """Detect anomalies in perception results"""
        # Check for sudden changes in detection counts
        if historical_data:
            avg_detections = np.mean([data['detection_count'] for data in historical_data])
            current_count = len(current_results.get('detections', []))

            if abs(current_count - avg_detections) > 2 * np.std([data['detection_count'] for data in historical_data]):
                return {'type': 'detection_count_anomaly', 'severity': 'medium'}

        # Check for consistency in detected classes
        current_classes = [det['label'] for det in current_results.get('detections', [])]
        if historical_data:
            historical_classes = [item for data in historical_data for item in data.get('classes', [])]
            new_classes_ratio = len(set(current_classes) - set(historical_classes)) / len(current_classes) if current_classes else 0

            if new_classes_ratio > 0.5:  # More than 50% new classes
                return {'type': 'class_distribution_anomaly', 'severity': 'high'}

        return {'type': 'normal', 'severity': 'none'}
```

## Key Takeaways

- Perception systems form the foundation of AI robot brains
- Computer vision enables environmental understanding and object recognition
- Multi-modal fusion combines different sensor inputs for robust perception
- GPU acceleration is crucial for real-time processing
- Quality assessment ensures reliable perception results
- NVIDIA Isaac provides optimized tools for perception tasks

## Next Steps

In the next chapter, we'll explore planning and decision-making systems, learning how AI robot brains transform perception data into intelligent actions and behaviors.