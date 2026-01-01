---
sidebar_position: 6
title: Advanced Communication Patterns
---

# Advanced Communication Patterns

In this chapter, we'll explore sophisticated communication patterns that go beyond the basic ROS 2 primitives. These patterns enable more complex robotic behaviors and efficient data handling in multi-robot systems.

## Custom Message Types

While ROS 2 provides many standard message types, you'll often need to create custom messages for your specific application.

### Creating Custom Messages

To create a custom message, define it in a `.msg` file:

```
# msg/RobotStatus.msg
string robot_name
bool is_active
float64 battery_level
int32 error_code
geometry_msgs/Pose current_pose
sensor_msgs/JointState joint_states
```

### Using Custom Messages

```python
# Python usage
from my_robot_msgs.msg import RobotStatus

class RobotStatusPublisher(Node):
    def __init__(self):
        super().__init__('robot_status_publisher')
        self.publisher = self.create_publisher(RobotStatus, 'robot_status', 10)

    def publish_status(self):
        msg = RobotStatus()
        msg.robot_name = "my_robot_01"
        msg.is_active = True
        msg.battery_level = 0.85
        msg.error_code = 0

        # Set pose and joint states...
        self.publisher.publish(msg)
```

## Complex Data Structures

### Message Composition

Complex data structures can be built by composing existing message types:

```python
# msg/RobotSystem.msg
string system_name
RobotStatus[] robot_statuses
sensor_msgs/BatteryState[] battery_states
geometry_msgs/TransformStamped[] transforms
diagnostic_msgs/DiagnosticArray diagnostics
```

### Array of Messages

Handling arrays of messages for multi-robot systems:

```python
from my_robot_msgs.msg import RobotStatusArray

class MultiRobotManager(Node):
    def __init__(self):
        super().__init__('multi_robot_manager')
        self.status_sub = self.create_subscription(
            RobotStatusArray, 'robot_status_array', self.status_array_callback, 10)

    def status_array_callback(self, msg):
        for robot_status in msg.robot_statuses:
            self.get_logger().info(
                f'Robot {robot_status.robot_name}: '
                f'Battery {robot_status.battery_level:.2f}, '
                f'Active: {robot_status.is_active}'
            )
```

## Communication Optimization

### Message Filtering and Throttling

Reduce bandwidth by filtering or throttling messages:

```python
from message_filters import Subscriber, TimeSynchronizer
from sensor_msgs.msg import Image, CameraInfo

class ImageProcessor(Node):
    def __init__(self):
        super().__init__('image_processor')

        # Subscribe to topics
        self.image_sub = Subscriber(self, Image, 'camera/image_raw')
        self.info_sub = Subscriber(self, CameraInfo, 'camera/camera_info')

        # Synchronize messages with time tolerance
        self.sync = TimeSynchronizer([self.image_sub, self.info_sub], 10)
        self.sync.registerCallback(self.image_callback)

    def image_callback(self, image_msg, info_msg):
        # Process synchronized image and camera info
        pass

# Alternative: Throttle messages to reduce processing
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy

class ThrottledSubscriber(Node):
    def __init__(self):
        super().__init__('throttled_subscriber')

        # Reduce message rate by using smaller queue depth
        qos_profile = QoSProfile(
            depth=1,  # Only keep latest message
            reliability=ReliabilityPolicy.BEST_EFFORT
        )

        self.sub = self.create_subscription(
            Image, 'high_freq_topic', self.callback, qos_profile)

    def callback(self, msg):
        # Process only the latest message
        pass
```

### Lazy Subscription

Only subscribe when there are active subscribers to your publications:

```python
class LazyPublisher(Node):
    def __init__(self):
        super().__init__('lazy_publisher')

        # Create publisher with custom callback for subscription changes
        self.publisher = self.create_publisher(
            Image, 'camera/image_raw', 10)

        # Create subscriber that only activates when needed
        self.subscription = None
        self.need_data = False

    def publisher_subscription_change_callback(self, subscription_count):
        if subscription_count > 0 and not self.subscription:
            # Someone is listening, start subscribing to camera
            self.subscription = self.create_subscription(
                Image, 'camera/raw', self.image_callback, 10)
            self.need_data = True
        elif subscription_count == 0 and self.subscription:
            # No one is listening, stop subscribing
            self.subscription.destroy()
            self.subscription = None
            self.need_data = False
```

## Multi-Node Communication Patterns

### Publisher-Subscriber with Multiple Publishers

Handling data from multiple similar nodes:

```python
class MultiSensorFusion(Node):
    def __init__(self):
        super().__init__('multi_sensor_fusion')

        # Store data from multiple sensors
        self.sensor_data = {}

        # Subscribe to sensors with different IDs
        for i in range(4):  # 4 sensors
            sensor_topic = f'sensor_{i}/data'
            self.create_subscription(
                SensorData, sensor_topic,
                lambda msg, id=i: self.sensor_callback(msg, id), 10)

    def sensor_callback(self, msg, sensor_id):
        self.sensor_data[sensor_id] = msg
        self.fuse_sensor_data()

    def fuse_sensor_data(self):
        # Combine data from all sensors
        if len(self.sensor_data) == 4:
            # Process all sensor data
            pass
```

### Master-Slave Architecture

Implementing a master-slave pattern for coordinated control:

```python
# Master node
class MasterController(Node):
    def __init__(self):
        super().__init__('master_controller')

        # Publishers for each slave
        self.slave_cmd_pubs = {}
        for i in range(3):  # 3 slave robots
            self.slave_cmd_pubs[i] = self.create_publisher(
                Twist, f'slave_{i}/cmd_vel', 10)

        # Subscribers for status from slaves
        for i in range(3):
            self.create_subscription(
                RobotStatus, f'slave_{i}/status',
                lambda msg, id=i: self.slave_status_callback(msg, id), 10)

        self.task_timer = self.create_timer(1.0, self.assign_tasks)

    def assign_tasks(self):
        # Coordinate tasks among slaves
        # Send commands to appropriate slaves
        pass

    def slave_status_callback(self, msg, slave_id):
        # Handle status from individual slave
        pass

# Slave node
class SlaveRobot(Node):
    def __init__(self, robot_id):
        super().__init__(f'slave_robot_{robot_id}')
        self.robot_id = robot_id

        # Subscribe to commands from master
        self.cmd_sub = self.create_subscription(
            Twist, f'slave_{robot_id}/cmd_vel', self.cmd_callback, 10)

        # Publish status to master
        self.status_pub = self.create_publisher(
            RobotStatus, f'slave_{robot_id}/status', 10)

    def cmd_callback(self, cmd_msg):
        # Execute command from master
        pass
```

## Advanced Service Patterns

### Batch Services

Services that handle multiple operations in a single call:

```python
# srv/BatchCommands.srv
BatchCommand[] commands
---
bool[] results
string error_message

# msg/BatchCommand.msg
string command_type  # "move", "sense", "calibrate", etc.
float64[] parameters
string target
```

```python
from my_robot_msgs.srv import BatchCommands

class BatchCommandServer(Node):
    def __init__(self):
        super().__init__('batch_command_server')
        self.srv = self.create_service(BatchCommands, 'batch_commands', self.batch_callback)

    def batch_callback(self, request, response):
        response.results = []

        for command in request.commands:
            try:
                result = self.execute_command(command)
                response.results.append(result)
            except Exception as e:
                response.results.append(False)
                response.error_message = str(e)
                return response

        return response

    def execute_command(self, command):
        # Execute individual command based on type
        if command.command_type == 'move':
            # Execute move command
            pass
        elif command.command_type == 'sense':
            # Execute sense command
            pass
        return True
```

### Service with Progress

Services that provide progress updates:

```python
# Instead of a regular service, use an action for progress reporting
from rclpy.action import ActionServer
from my_robot_msgs.action import Calibration

class CalibrationActionServer(Node):
    def __init__(self):
        super().__init__('calibration_server')
        self._action_server = ActionServer(
            self,
            Calibration,
            'calibrate_sensors',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback)

    def execute_callback(self, goal_handle):
        feedback_msg = Calibration.Feedback()

        # Perform calibration steps
        steps = ['initialize', 'collect_data', 'analyze', 'apply_calibration']

        for i, step in enumerate(steps):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return Calibration.Result(success=False)

            # Perform calibration step
            self.perform_calibration_step(step)

            # Update feedback
            feedback_msg.current_step = step
            feedback_msg.progress = (i + 1) / len(steps) * 100.0
            goal_handle.publish_feedback(feedback_msg)

        goal_handle.succeed()
        result = Calibration.Result()
        result.success = True
        return result
```

## Performance Optimization

### Shared Memory

For high-frequency data, consider using shared memory:

```python
import numpy as np
from sensor_msgs.msg import Image
import cv2

class SharedMemoryPublisher(Node):
    def __init__(self):
        super().__init__('shared_memory_publisher')
        self.image_pub = self.create_publisher(Image, 'shared_image', 1)

        # For high-performance scenarios, consider using shared memory
        # libraries like shm or memory-mapped files for large data
        self.image_buffer = None

    def publish_large_image(self, image_data):
        # Convert to ROS Image message
        img_msg = Image()
        img_msg.height = image_data.shape[0]
        img_msg.width = image_data.shape[1]
        img_msg.encoding = "bgr8"
        img_msg.is_bigendian = False
        img_msg.step = image_data.shape[1] * 3
        img_msg.data = image_data.tobytes()

        self.image_pub.publish(img_msg)
```

### Message Compression

For bandwidth-limited scenarios, implement compression:

```python
import cv2
import numpy as np
from std_msgs.msg import UInt8MultiArray

class CompressedImagePublisher(Node):
    def __init__(self):
        super().__init__('compressed_image_publisher')
        self.compressed_pub = self.create_publisher(UInt8MultiArray, 'compressed_image', 1)

    def publish_compressed(self, image):
        # Compress image using OpenCV
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]  # 50% quality
        result, encoded_image = cv2.imencode('.jpg', image, encode_param)

        if result:
            # Publish compressed data
            compressed_msg = UInt8MultiArray()
            compressed_msg.data = encoded_image.tobytes()
            self.compressed_pub.publish(compressed_msg)
```

## Real-time Communication Considerations

### Real-time Publisher

For time-critical applications, use real-time capabilities:

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

class RealTimePublisher(Node):
    def __init__(self):
        super().__init__('realtime_publisher')

        # QoS for real-time communication
        realtime_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,  # Minimal queue to reduce latency
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            deadline=(0, 100000000),  # 100ms deadline
            lifespan=(0, 100000000),  # 100ms lifespan
        )

        self.publisher = self.create_publisher(Twist, 'realtime_cmd', realtime_qos)

        # High-frequency timer for real-time control
        self.timer = self.create_timer(0.01, self.control_callback)  # 10ms period

    def control_callback(self):
        # Generate real-time control commands
        cmd = Twist()
        # Fill in command...
        self.publisher.publish(cmd)
```

## Practical Example: Multi-Robot Coordination System

Let's build a comprehensive example that combines multiple advanced patterns:

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import String
from std_srvs.srv import Trigger
from my_robot_msgs.msg import RobotStatusArray
from my_robot_msgs.srv import TaskAssignment

class MultiRobotCoordinator(Node):
    def __init__(self):
        super().__init__('multi_robot_coordinator')

        # Publishers
        self.task_pub = self.create_publisher(String, 'task_queue', 10)
        self.status_pub = self.create_publisher(RobotStatusArray, 'fleet_status', 10)

        # Subscribers for robot status
        self.robot_statuses = {}
        for i in range(3):  # 3 robots in fleet
            self.create_subscription(
                String, f'robot_{i}/status',
                lambda msg, id=i: self.robot_status_callback(msg, id),
                QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))

        # Service for task assignment
        self.task_assignment_srv = self.create_service(
            TaskAssignment, 'assign_task', self.assign_task_callback)

        # Service for fleet management
        self.fleet_control_srv = self.create_service(
            Trigger, 'fleet_emergency_stop', self.emergency_stop_callback)

        # Timer for coordination logic
        self.coordination_timer = self.create_timer(0.5, self.coordination_loop)

        self.get_logger().info('Multi-Robot Coordinator initialized')

    def robot_status_callback(self, msg, robot_id):
        """Handle status updates from individual robots"""
        self.robot_statuses[robot_id] = {
            'status': msg.data,
            'timestamp': self.get_clock().now()
        }

    def assign_task_callback(self, request, response):
        """Assign tasks to robots based on their current status"""
        # Find available robot
        available_robot = None
        for robot_id, status_info in self.robot_statuses.items():
            if status_info['status'] == 'idle':
                available_robot = robot_id
                break

        if available_robot is not None:
            # Publish task to specific robot
            task_msg = String()
            task_msg.data = f"TASK:{request.task_description}"
            task_pub = self.create_publisher(String, f'robot_{available_robot}/task', 10)
            task_pub.publish(task_msg)

            response.success = True
            response.assigned_robot = f'robot_{available_robot}'
        else:
            response.success = False
            response.assigned_robot = 'none'

        return response

    def emergency_stop_callback(self, request, response):
        """Send emergency stop to all robots"""
        for i in range(3):
            stop_pub = self.create_publisher(Twist, f'robot_{i}/cmd_vel', 1)
            stop_cmd = Twist()
            stop_pub.publish(stop_cmd)  # Zero velocity = stop

        response.success = True
        return response

    def coordination_loop(self):
        """Main coordination logic"""
        # Publish fleet status
        status_array = RobotStatusArray()
        for robot_id, status_info in self.robot_statuses.items():
            # Create robot status message
            pass

        self.status_pub.publish(status_array)

def main(args=None):
    rclpy.init(args=args)
    coordinator = MultiRobotCoordinator()

    try:
        rclpy.spin(coordinator)
    except KeyboardInterrupt:
        pass
    finally:
        coordinator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Communication Security

### Secure Communication

Implement security measures for sensitive robotic systems:

```python
# While ROS 2 doesn't have built-in security, you can implement:
# - Message signing for authenticity
# - Encryption for sensitive data
# - Access control lists

class SecureNode(Node):
    def __init__(self):
        super().__init__('secure_node')

        # Use secure topics for sensitive data
        # This is a conceptual example - actual implementation would require
        # additional security infrastructure
        self.secure_pub = self.create_publisher(String, 'secure_commands', 10)
        self.secure_sub = self.create_subscription(
            String, 'secure_feedback', self.secure_callback, 10)

    def secure_callback(self, msg):
        # Verify message authenticity before processing
        # Decrypt if necessary
        # Process secure command
        pass
```

## Key Takeaways

- Custom messages enable domain-specific communication
- Complex data structures can be built by composing existing types
- Performance optimization includes filtering, throttling, and lazy subscription
- Multi-node patterns include master-slave and coordination architectures
- Advanced services can handle batch operations and progress reporting
- Real-time considerations require appropriate QoS settings
- Security should be considered for sensitive applications

## Next Steps

In the final chapter of this module, we'll explore real-world applications and case studies, bringing together all the concepts we've learned to build complete robotic communication systems.