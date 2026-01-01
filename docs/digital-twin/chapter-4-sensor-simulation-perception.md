---
sidebar_position: 4
title: Sensor Simulation and Perception
---

# Sensor Simulation and Perception

In this chapter, we'll explore how to simulate various sensors in Gazebo and create realistic perception systems for digital twins. We'll cover cameras, lidars, IMUs, GPS, and other sensors that enable robots to perceive their environment in simulation.

## Understanding Sensor Simulation

Sensor simulation in Gazebo aims to replicate the behavior of physical sensors as accurately as possible. This includes:

- **Physical properties**: Range, resolution, field of view, noise characteristics
- **Environmental factors**: Lighting conditions, weather, occlusions
- **Sensor limitations**: Accuracy, latency, refresh rates
- **Data formats**: Output in standard ROS message formats

## Camera Simulation

### Basic Camera Configuration

```xml
<!-- In URDF/Gazebo tags -->
<gazebo reference="camera_link">
  <sensor type="camera" name="camera1">
    <update_rate>30.0</update_rate>
    <camera name="head">
      <horizontal_fov>1.047</horizontal_fov> <!-- 60 degrees in radians -->
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>100</far>
      </clip>
      <noise>
        <type>gaussian</type>
        <mean>0.0</mean>
        <stddev>0.007</stddev>
      </noise>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <frame_name>camera_optical_frame</frame_name>
      <min_depth>0.1</min_depth>
      <max_depth>100.0</max_depth>
    </plugin>
  </sensor>
</gazebo>
```

### Depth Camera Simulation

```xml
<gazebo reference="depth_camera_link">
  <sensor type="depth" name="depth_camera">
    <update_rate>30.0</update_rate>
    <camera name="depth_head">
      <horizontal_fov>1.047</horizontal_fov>
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>10.0</far>
      </clip>
    </camera>
    <plugin name="depth_camera_controller" filename="libgazebo_ros_openni_kinect.so">
      <baseline>0.2</baseline>
      <alwaysOn>true</alwaysOn>
      <updateRate>30.0</updateRate>
      <cameraName>camera</cameraName>
      <imageTopicName>rgb/image_raw</imageTopicName>
      <depthImageTopicName>depth/image_raw</depthImageTopicName>
      <pointCloudTopicName>depth/points</pointCloudTopicName>
      <cameraInfoTopicName>rgb/camera_info</cameraInfoTopicName>
      <depthImageCameraInfoTopicName>depth/camera_info</depthImageCameraInfoTopicName>
      <frameName>depth_camera_optical_frame</frameName>
      <pointCloudCutoff>0.1</pointCloudCutoff>
      <pointCloudCutoffMax>10.0</pointCloudCutoffMax>
      <distortion_k1>0.0</distortion_k1>
      <distortion_k2>0.0</distortion_k2>
      <distortion_k3>0.0</distortion_k3>
      <distortion_t1>0.0</distortion_t1>
      <distortion_t2>0.0</distortion_t2>
      <CxPrime>0</CxPrime>
      <Cx>320.5</Cx>
      <Cy>240.5</Cy>
      <focalLength>320.0</focalLength>
      <hackBaseline>0</hackBaseline>
    </plugin>
  </sensor>
</gazebo>
```

## LIDAR Simulation

### 2D LIDAR (Hokuyo-like)

```xml
<gazebo reference="laser_link">
  <sensor type="ray" name="laser_scan">
    <pose>0 0 0 0 0 0</pose>
    <visualize>true</visualize>
    <update_rate>40</update_rate>
    <ray>
      <scan>
        <horizontal>
          <samples>720</samples>
          <resolution>1</resolution>
          <min_angle>-3.14159</min_angle> <!-- -180 degrees -->
          <max_angle>3.14159</max_angle>   <!-- 180 degrees -->
        </horizontal>
      </scan>
      <range>
        <min>0.10</min>
        <max>30.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <plugin name="laser_scan_controller" filename="libgazebo_ros_ray_sensor.so">
      <ros>
        <namespace>/laser</namespace>
        <remapping>~/out:=scan</remapping>
      </ros>
      <output_type>sensor_msgs/LaserScan</output_type>
      <frame_name>laser_link</frame_name>
    </plugin>
  </sensor>
</gazebo>
```

### 3D LIDAR (Velodyne-like)

```xml
<gazebo reference="velodyne_link">
  <sensor type="ray" name="velodyne-VLP-16">
    <pose>0 0 0 0 0 0</pose>
    <visualize>false</visualize>
    <update_rate>10</update_rate>
    <ray>
      <scan>
        <horizontal>
          <samples>1800</samples>
          <resolution>1</resolution>
          <min_angle>-3.14159</min_angle>
          <max_angle>3.14159</max_angle>
        </horizontal>
        <vertical>
          <samples>16</samples>
          <resolution>1</resolution>
          <min_angle>-0.261799</min_angle> <!-- -15 degrees -->
          <max_angle>0.261799</max_angle>   <!-- 15 degrees -->
        </vertical>
      </scan>
      <range>
        <min>0.1</min>
        <max>100.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <plugin name="velodyne_controller" filename="libgazebo_ros_velodyne_laser.so">
      <topicName>velodyne_points</topicName>
      <frameName>velodyne_link</frameName>
      <min_range>0.9</min_range>
      <max_range>130.0</max_range>
      <gaussian_noise>0.008</gaussian_noise>
    </plugin>
  </sensor>
</gazebo>
```

## IMU Simulation

```xml
<gazebo reference="imu_link">
  <sensor type="imu" name="imu_sensor">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <visualize>false</visualize>
    <imu>
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
            <bias_mean>0.0000075</bias_mean>
            <bias_stddev>0.0000008</bias_stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
            <bias_mean>0.0000075</bias_mean>
            <bias_stddev>0.0000008</bias_stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
            <bias_mean>0.0000075</bias_mean>
            <bias_stddev>0.0000008</bias_stddev>
          </noise>
        </z>
      </angular_velocity>
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
            <bias_mean>0.1</bias_mean>
            <bias_stddev>0.001</bias_stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
            <bias_mean>0.1</bias_mean>
            <bias_stddev>0.001</bias_stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
            <bias_mean>0.1</bias_mean>
            <bias_stddev>0.001</bias_stddev>
          </noise>
        </z>
      </linear_acceleration>
    </imu>
    <plugin name="imu_plugin" filename="libgazebo_ros_imu.so">
      <ros>
        <namespace>/imu</namespace>
        <remapping>~/out:=data</remapping>
      </ros>
      <frame_name>imu_link</frame_name>
      <body_name>imu_link</body_name>
      <update_rate>100</update_rate>
      <gaussian_noise>0.001</gaussian_noise>
    </plugin>
  </sensor>
</gazebo>
```

## GPS Simulation

```xml
<gazebo reference="gps_link">
  <sensor type="gps" name="navsat">
    <always_on>true</always_on>
    <update_rate>4</update_rate>
    <plugin name="gps_plugin" filename="libgazebo_ros_navsat.so">
      <ros>
        <namespace>/gps</namespace>
      </ros>
      <frame_name>gps_link</frame_name>
      <update_rate>4</update_rate>
      <fix_topic>fix</fix_topic>
      <status_topic>status</status_topic>
      <time_ref_topic>time_reference</time_ref_topic>
      <latitude>49.9213</latitude>
      <longitude>8.8985</longitude>
      <elevation>0.0</elevation>
      <position_error>0.1</position_error>
    </plugin>
  </sensor>
</gazebo>
```

## Force/Torque Sensor

```xml
<gazebo>
  <sensor name="ft_sensor" type="force_torque">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <force_torque>
      <frame>child</frame>
      <measure_direction>child_to_parent</measure_direction>
    </force_torque>
    <plugin name="ft_sensor_plugin" filename="libgazebo_ros_ft_sensor.so">
      <frame_name>ft_sensor_link</frame_name>
      <topic>ft_sensor_topic</topic>
    </plugin>
  </sensor>
</gazebo>
```

## Multi-Sensor Integration Example

Here's a complete example integrating multiple sensors on a robot:

```xml
<?xml version="1.0"?>
<robot name="sensor_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Base link -->
  <link name="base_link">
    <inertial>
      <mass value="10.0" />
      <origin xyz="0 0 0.1" rpy="0 0 0" />
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.2" />
    </inertial>
    <visual>
      <origin xyz="0 0 0.1" rpy="0 0 0" />
      <geometry>
        <box size="0.5 0.3 0.2" />
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0.1" rpy="0 0 0" />
      <geometry>
        <box size="0.5 0.3 0.2" />
      </geometry>
    </collision>
  </link>

  <!-- IMU -->
  <joint name="imu_joint" type="fixed">
    <parent link="base_link" />
    <child link="imu_link" />
    <origin xyz="0 0 0.1" rpy="0 0 0" />
  </joint>
  <link name="imu_link" />

  <!-- Camera -->
  <joint name="camera_joint" type="fixed">
    <parent link="base_link" />
    <child link="camera_link" />
    <origin xyz="0.2 0 0.15" rpy="0 0 0" />
  </joint>
  <link name="camera_link" />

  <!-- 2D LIDAR -->
  <joint name="laser_joint" type="fixed">
    <parent link="base_link" />
    <child link="laser_link" />
    <origin xyz="0.15 0 0.2" rpy="0 0 0" />
  </joint>
  <link name="laser_link" />

  <!-- GPS -->
  <joint name="gps_joint" type="fixed">
    <parent link="base_link" />
    <child link="gps_link" />
    <origin xyz="0 0 0.3" rpy="0 0 0" />
  </joint>
  <link name="gps_link" />

  <!-- Gazebo sensor definitions -->
  <gazebo reference="imu_link">
    <sensor type="imu" name="imu_sensor">
      <always_on>true</always_on>
      <update_rate>100</update_rate>
      <visualize>false</visualize>
      <imu>
        <angular_velocity>
          <x><noise type="gaussian"><mean>0.0</mean><stddev>2e-4</stddev></noise></x>
          <y><noise type="gaussian"><mean>0.0</mean><stddev>2e-4</stddev></noise></y>
          <z><noise type="gaussian"><mean>0.0</mean><stddev>2e-4</stddev></noise></z>
        </angular_velocity>
        <linear_acceleration>
          <x><noise type="gaussian"><mean>0.0</mean><stddev>1.7e-2</stddev></noise></x>
          <y><noise type="gaussian"><mean>0.0</mean><stddev>1.7e-2</stddev></noise></y>
          <z><noise type="gaussian"><mean>0.0</mean><stddev>1.7e-2</stddev></noise></z>
        </linear_acceleration>
      </imu>
      <plugin name="imu_plugin" filename="libgazebo_ros_imu.so">
        <frame_name>imu_link</frame_name>
        <body_name>imu_link</body_name>
        <update_rate>100</update_rate>
      </plugin>
    </sensor>
  </gazebo>

  <gazebo reference="camera_link">
    <sensor type="camera" name="camera1">
      <update_rate>30.0</update_rate>
      <camera name="head">
        <horizontal_fov>1.047</horizontal_fov>
        <image>
          <width>640</width>
          <height>480</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.1</near>
          <far>100</far>
        </clip>
      </camera>
      <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
        <frame_name>camera_optical_frame</frame_name>
      </plugin>
    </sensor>
  </gazebo>

  <gazebo reference="laser_link">
    <sensor type="ray" name="laser_scan">
      <visualize>true</visualize>
      <update_rate>40</update_rate>
      <ray>
        <scan>
          <horizontal>
            <samples>720</samples>
            <resolution>1</resolution>
            <min_angle>-3.14159</min_angle>
            <max_angle>3.14159</max_angle>
          </horizontal>
        </scan>
        <range>
          <min>0.10</min>
          <max>30.0</max>
          <resolution>0.01</resolution>
        </range>
      </ray>
      <plugin name="laser_scan_controller" filename="libgazebo_ros_ray_sensor.so">
        <output_type>sensor_msgs/LaserScan</output_type>
        <frame_name>laser_link</frame_name>
      </plugin>
    </sensor>
  </gazebo>

  <gazebo reference="gps_link">
    <sensor type="gps" name="navsat">
      <always_on>true</always_on>
      <update_rate>4</update_rate>
      <plugin name="gps_plugin" filename="libgazebo_ros_navsat.so">
        <frame_name>gps_link</frame_name>
        <latitude>49.9213</latitude>
        <longitude>8.8985</longitude>
        <elevation>0.0</elevation>
      </plugin>
    </sensor>
  </gazebo>
</robot>
```

## Perception Processing in ROS 2

### Camera Processing Node

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np

class CameraProcessor(Node):
    def __init__(self):
        super().__init__('camera_processor')

        # Create CV bridge for image conversion
        self.cv_bridge = CvBridge()

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.info_sub = self.create_subscription(
            CameraInfo, '/camera/camera_info', self.info_callback, 10)

        # Publishers
        self.processed_pub = self.create_publisher(Image, '/camera/processed', 10)

        # Camera parameters
        self.camera_matrix = None
        self.distortion_coeffs = None

    def info_callback(self, msg):
        """Receive camera calibration info"""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.distortion_coeffs = np.array(msg.d)

    def image_callback(self, msg):
        """Process incoming camera images"""
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")

            # Apply processing (example: edge detection)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            # Convert back to ROS Image
            processed_msg = self.cv_bridge.cv2_to_imgmsg(edges, "mono8")
            processed_msg.header = msg.header

            # Publish processed image
            self.processed_pub.publish(processed_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def detect_objects(self, cv_image):
        """Example object detection function"""
        # Convert to HSV for color-based detection
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Define range for red color
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)

        lower_red = np.array([170, 50, 50])
        upper_red = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red, upper_red)

        mask = mask1 + mask2

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding boxes
        result = cv_image.copy()
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Filter small contours
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)

        return result
```

### LIDAR Processing Node

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point
import numpy as np

class LIDARProcessor(Node):
    def __init__(self):
        super().__init__('lidar_processor')

        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)

        # Publishers
        self.obstacle_pub = self.create_publisher(Point, '/obstacle_detected', 10)

        # Parameters
        self.declare_parameter('safety_distance', 0.5)
        self.declare_parameter('min_obstacle_size', 0.2)

    def scan_callback(self, msg):
        """Process LIDAR scan data"""
        # Get parameter values
        safety_distance = self.get_parameter('safety_distance').value
        min_obstacle_size = self.get_parameter('min_obstacle_size').value

        # Convert scan to Cartesian coordinates
        angles = np.array([msg.angle_min + i * msg.angle_increment
                          for i in range(len(msg.ranges))])
        ranges = np.array(msg.ranges)

        # Filter valid ranges
        valid_mask = (ranges > msg.range_min) & (ranges < msg.range_max) & (ranges < safety_distance)
        valid_angles = angles[valid_mask]
        valid_ranges = ranges[valid_mask]

        if len(valid_ranges) > 0:
            # Convert to Cartesian
            x_coords = valid_ranges * np.cos(valid_angles)
            y_coords = valid_ranges * np.sin(valid_angles)

            # Cluster points to find obstacles
            obstacles = self.cluster_obstacles(x_coords, y_coords, min_obstacle_size)

            # Publish first detected obstacle
            if obstacles:
                obstacle_msg = Point()
                obstacle_msg.x = obstacles[0][0]  # x coordinate
                obstacle_msg.y = obstacles[0][1]  # y coordinate
                obstacle_msg.z = 0.0
                self.obstacle_pub.publish(obstacle_msg)

    def cluster_obstacles(self, x_coords, y_coords, min_distance):
        """Cluster nearby points into obstacles"""
        if len(x_coords) == 0:
            return []

        points = np.column_stack((x_coords, y_coords))
        obstacles = []
        used = [False] * len(points)

        for i, point in enumerate(points):
            if used[i]:
                continue

            cluster = [point]
            used[i] = True

            # Find nearby points
            for j, other_point in enumerate(points):
                if used[j]:
                    continue

                distance = np.sqrt((point[0] - other_point[0])**2 + (point[1] - other_point[1])**2)
                if distance < min_distance:
                    cluster.append(other_point)
                    used[j] = True

            # Calculate cluster center
            cluster_center = np.mean(cluster, axis=0)
            obstacles.append(cluster_center)

        return obstacles
```

## Sensor Fusion

### Multi-Sensor Data Integration

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image, Imu
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
import numpy as np
from scipy.spatial.transform import Rotation as R

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion')

        # Subscribers for different sensors
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.imu_sub = self.create_subscription(Imu, '/imu/data', self.imu_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # Publishers
        self.fused_pose_pub = self.create_publisher(PoseStamped, '/fused_pose', 10)
        self.environment_map_pub = self.create_publisher(OccupancyGrid, '/environment_map', 10)

        # Sensor data storage
        self.lidar_data = None
        self.imu_data = None
        self.odom_data = None

        # Fusion parameters
        self.lidar_weight = 0.3
        self.imu_weight = 0.4
        self.odom_weight = 0.3

    def lidar_callback(self, msg):
        self.lidar_data = msg

    def imu_callback(self, msg):
        self.imu_data = msg

    def odom_callback(self, msg):
        self.odom_data = msg

    def sensor_fusion_update(self):
        """Combine sensor data for improved state estimation"""
        if not all([self.lidar_data, self.imu_data, self.odom_data]):
            return

        # Extract pose information from odometry
        odom_pose = self.odom_data.pose.pose

        # Extract orientation from IMU
        imu_quat = [
            self.imu_data.orientation.x,
            self.imu_data.orientation.y,
            self.imu_data.orientation.z,
            self.imu_data.orientation.w
        ]

        # Create fused pose (simplified - in practice, use Kalman filter or similar)
        fused_pose = PoseStamped()
        fused_pose.header.stamp = self.get_clock().now().to_msg()
        fused_pose.header.frame_id = 'map'

        # Weighted average of position (from odom) and orientation (from IMU)
        fused_pose.pose.position = odom_pose.position
        fused_pose.pose.orientation = self.imu_data.orientation

        # Publish fused pose
        self.fused_pose_pub.publish(fused_pose)

        # Create environment map from LIDAR data
        if self.lidar_data:
            self.create_environment_map(self.lidar_data)

    def create_environment_map(self, lidar_msg):
        """Create occupancy grid from LIDAR data"""
        # Convert LIDAR ranges to Cartesian coordinates
        angles = np.array([lidar_msg.angle_min + i * lidar_msg.angle_increment
                          for i in range(len(lidar_msg.ranges))])
        ranges = np.array(lidar_msg.ranges)

        # Filter valid measurements
        valid_mask = (ranges > lidar_msg.range_min) & (ranges < lidar_msg.range_max)
        valid_angles = angles[valid_mask]
        valid_ranges = ranges[valid_mask]

        # Convert to Cartesian coordinates
        x_coords = valid_ranges * np.cos(valid_angles)
        y_coords = valid_ranges * np.sin(valid_angles)

        # Create occupancy grid (simplified)
        map_msg = OccupancyGrid()
        map_msg.header.stamp = lidar_msg.header.stamp
        map_msg.header.frame_id = 'map'
        map_msg.info.resolution = 0.1  # 10cm resolution
        map_msg.info.width = 200  # 20m x 20m map
        map_msg.info.height = 200
        map_msg.info.origin.position.x = -10.0
        map_msg.info.origin.position.y = -10.0

        # Initialize map (simplified - in practice, use proper ray casting)
        map_data = [-1] * (map_msg.info.width * map_msg.info.height)  # -1 = unknown
        self.update_map_with_lidar_scan(map_data, x_coords, y_coords, map_msg)

        map_msg.data = map_data
        self.environment_map_pub.publish(map_msg)

    def update_map_with_lidar_scan(self, map_data, x_coords, y_coords, map_info):
        """Update occupancy grid with LIDAR scan"""
        for x, y in zip(x_coords, y_coords):
            # Convert world coordinates to map indices
            map_x = int((x - map_info.info.origin.position.x) / map_info.info.resolution)
            map_y = int((y - map_info.info.origin.position.y) / map_info.info.resolution)

            # Check bounds
            if 0 <= map_x < map_info.info.width and 0 <= map_y < map_info.info.height:
                index = map_y * map_info.info.width + map_x
                if 0 <= index < len(map_data):
                    map_data[index] = 100  # 100 = occupied
```

## Advanced Perception Techniques

### SLAM Integration

```python
class SLAMNode(Node):
    def __init__(self):
        super().__init__('slam_node')

        # Subscribers
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # Publishers
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # SLAM algorithm parameters
        self.scan_matcher = ScanMatcher()  # Simplified representation
        self.map_resolution = 0.05
        self.map_size = 1000  # 50m x 50m at 5cm resolution

    def lidar_callback(self, scan_msg):
        """Process LIDAR data for SLAM"""
        # Convert scan to appropriate format for SLAM algorithm
        scan_points = self.convert_scan_to_points(scan_msg)

        # Perform scan matching to estimate pose
        estimated_pose = self.scan_matcher.match_scan(
            scan_points, self.current_map, self.previous_pose)

        # Update pose estimate
        self.current_pose = estimated_pose

        # Update map with new scan
        self.update_map(scan_points, estimated_pose)

        # Publish map and transform
        self.publish_map()
        self.publish_transform()

    def convert_scan_to_points(self, scan_msg):
        """Convert LaserScan message to point cloud"""
        angles = np.array([scan_msg.angle_min + i * scan_msg.angle_increment
                          for i in range(len(scan_msg.ranges))])
        ranges = np.array(scan_msg.ranges)

        valid_mask = (ranges > scan_msg.range_min) & (ranges < scan_msg.range_max)
        valid_angles = angles[valid_mask]
        valid_ranges = ranges[valid_mask]

        x_coords = valid_ranges * np.cos(valid_angles)
        y_coords = valid_ranges * np.sin(valid_angles)

        return np.column_stack((x_coords, y_coords))
```

## Sensor Calibration

### Camera Calibration in Simulation

```python
class CameraCalibrationNode(Node):
    def __init__(self):
        super().__init__('camera_calibration')

        # For simulation, we can directly access the ideal parameters
        # In real systems, these would be determined through calibration
        self.camera_matrix = np.array([
            [500.0, 0.0, 320.0],  # fx, 0, cx
            [0.0, 500.0, 240.0],  # 0, fy, cy
            [0.0, 0.0, 1.0]       # 0, 0, 1
        ])

        self.distortion_coeffs = np.array([0.1, -0.2, 0.0, 0.0, 0.0])  # k1, k2, p1, p2, k3

    def undistort_image(self, image):
        """Remove distortion from image"""
        h, w = image.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.distortion_coeffs, (w, h), 1, (w, h))

        undistorted = cv2.undistort(
            image, self.camera_matrix, self.distortion_coeffs, None, new_camera_matrix)

        return undistorted
```

## Performance Optimization

### Efficient Sensor Data Processing

```python
class OptimizedSensorProcessor(Node):
    def __init__(self):
        super().__init__('optimized_sensor_processor')

        # Use threading for CPU-intensive processing
        import threading
        from collections import deque

        self.processing_queue = deque(maxlen=5)
        self.processing_thread = threading.Thread(target=self.processing_worker)
        self.processing_lock = threading.Lock()
        self.processing_thread.start()

        # Throttle sensor subscriptions
        from message_filters import Subscriber, TimeSynchronizer
        self.image_sub = Subscriber(self, Image, '/camera/image_raw')
        self.scan_sub = Subscriber(self, LaserScan, '/scan')

        # Synchronize with time tolerance
        self.sync = TimeSynchronizer([self.image_sub, self.scan_sub], 10)
        self.sync.registerCallback(self.synchronized_callback)

    def synchronized_callback(self, image_msg, scan_msg):
        """Process synchronized sensor data"""
        with self.processing_lock:
            if len(self.processing_queue) < 5:  # Limit queue size
                self.processing_queue.append((image_msg, scan_msg))

    def processing_worker(self):
        """Background processing thread"""
        while rclpy.ok():
            with self.processing_lock:
                if self.processing_queue:
                    image_msg, scan_msg = self.processing_queue.popleft()
                else:
                    image_msg, scan_msg = None, None

            if image_msg is not None and scan_msg is not None:
                # Process sensor data
                result = self.process_sensors(image_msg, scan_msg)
                # Publish results...

            time.sleep(0.001)  # Small sleep to prevent busy waiting
```

## Key Takeaways

- Sensor simulation in Gazebo replicates physical sensor behavior with realistic noise and limitations
- Multiple sensor types (camera, LIDAR, IMU, GPS) provide complementary perception capabilities
- Proper sensor configuration includes realistic noise models and physical parameters
- Sensor fusion combines data from multiple sensors for improved accuracy
- Efficient processing techniques are essential for real-time perception
- Calibration parameters ensure accurate sensor measurements

## Next Steps

In the next chapter, we'll explore Unity robotics simulation, learning how to create advanced, photorealistic simulation environments for complex perception and AI tasks.