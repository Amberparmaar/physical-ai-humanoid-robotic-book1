---
sidebar_position: 7
title: Real-world Applications and Case Studies
---

# Real-world Applications and Case Studies

In this final chapter of Module 1, we'll explore real-world applications of ROS 2 communication patterns in actual robotic systems. Through detailed case studies, we'll see how the concepts we've learned come together in practical implementations.

## Case Study 1: Autonomous Mobile Robot (AMR) Fleet

Let's examine how a fleet of autonomous mobile robots in a warehouse environment uses ROS 2 communication patterns.

### System Architecture

```python
# Fleet management system
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import String
from my_fleet_msgs.msg import RobotStatus, TaskAssignment
from my_fleet_msgs.srv import RequestTask, FleetStatus

class FleetManager(Node):
    def __init__(self):
        super().__init__('fleet_manager')

        # Publishers for fleet coordination
        self.task_queue_pub = self.create_publisher(TaskAssignment, 'task_queue', 100)
        self.fleet_status_pub = self.create_publisher(RobotStatus, 'fleet_status', 10)

        # Subscribers for robot feedback
        self.robot_statuses = {}
        for i in range(10):  # 10 robots in fleet
            self.create_subscription(
                RobotStatus, f'robot_{i}/status',
                lambda msg, id=i: self.robot_status_callback(msg, id),
                QoSProfile(depth=10, durability=1))

        # Service for task assignment
        self.task_service = self.create_service(
            RequestTask, 'request_task', self.handle_task_request)

        # Timer for fleet optimization
        self.optimization_timer = self.create_timer(5.0, self.optimize_fleet)

    def robot_status_callback(self, msg, robot_id):
        """Update robot status and availability"""
        self.robot_statuses[robot_id] = msg
        self.get_logger().info(f'Robot {robot_id} status: {msg.state}')

    def handle_task_request(self, request, response):
        """Assign tasks based on robot availability and location"""
        # Find optimal robot for the task
        best_robot = self.find_best_robot_for_task(request.task)

        if best_robot is not None:
            # Assign task to robot
            task_msg = TaskAssignment()
            task_msg.robot_id = best_robot
            task_msg.task = request.task
            self.task_queue_pub.publish(task_msg)

            response.success = True
            response.assigned_robot = best_robot
        else:
            response.success = False
            response.assigned_robot = -1

        return response

    def find_best_robot_for_task(self, task):
        """Find the most suitable robot for a given task"""
        available_robots = [
            robot_id for robot_id, status in self.robot_statuses.items()
            if status.state == 'available'
        ]

        if not available_robots:
            return None

        # Simple selection based on proximity (in real system, use path planning)
        return min(available_robots)  # For demo, just pick first available

    def optimize_fleet(self):
        """Optimize fleet operations based on current status"""
        active_robots = sum(1 for status in self.robot_statuses.values()
                          if status.state == 'active')
        available_robots = sum(1 for status in self.robot_statuses.values()
                             if status.state == 'available')

        self.get_logger().info(
            f'Fleet status: {active_robots} active, {available_robots} available'
        )
```

### Robot Node Implementation

```python
class AMRobot(Node):
    def __init__(self, robot_id):
        super().__init__(f'am_robot_{robot_id}')
        self.robot_id = robot_id
        self.current_state = 'idle'
        self.current_task = None

        # Publishers
        self.status_pub = self.create_publisher(RobotStatus, f'robot_{robot_id}/status', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, f'robot_{robot_id}/cmd_vel', 10)

        # Subscribers
        self.task_sub = self.create_subscription(
            TaskAssignment, f'robot_{robot_id}/task_assignment',
            self.task_assignment_callback, 10)

        # Navigation action client
        self.nav_client = ActionClient(self, NavigateToPose, f'robot_{robot_id}/navigate_to_pose')

        # Timer for status updates
        self.status_timer = self.create_timer(1.0, self.publish_status)

    def task_assignment_callback(self, msg):
        """Handle assigned tasks"""
        if self.current_state == 'idle':
            self.current_task = msg.task
            self.current_state = 'executing'
            self.execute_task()

    def execute_task(self):
        """Execute the assigned task"""
        if self.current_task.task_type == 'transport':
            # Navigate to pickup location
            self.navigate_to_pose(self.current_task.pickup_pose)
        elif self.current_task.task_type == 'delivery':
            # Navigate to delivery location
            self.navigate_to_pose(self.current_task.delivery_pose)

    def navigate_to_pose(self, pose):
        """Navigate to specified pose using action"""
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Navigation server not available')
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose
        self.nav_client.send_goal_async(goal_msg)

    def publish_status(self):
        """Publish current robot status"""
        status_msg = RobotStatus()
        status_msg.robot_id = self.robot_id
        status_msg.state = self.current_state
        status_msg.battery_level = self.get_battery_level()
        status_msg.current_pose = self.get_current_pose()
        self.status_pub.publish(status_msg)

    def get_battery_level(self):
        # Simulated battery level
        return 0.85

    def get_current_pose(self):
        # Simulated pose
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        return pose.pose
```

## Case Study 2: Humanoid Robot Control System

Humanoid robots require sophisticated communication between multiple subsystems to achieve coordinated movement and interaction.

### Centralized Control Architecture

```python
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import WrenchStamped
from my_humanoid_msgs.msg import BalanceState, MotorCommand, SensorData

class HumanoidController(Node):
    def __init__(self):
        super().__init__('humanoid_controller')

        # Subsystems
        self.walking_controller = WalkingController(self)
        self.arm_controller = ArmController(self)
        self.balance_controller = BalanceController(self)

        # Publishers for motor commands
        self.motor_cmd_pub = self.create_publisher(MotorCommand, 'motor_commands', 100)

        # Subscribers for sensor data
        self.joint_state_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, 10)
        self.force_sub = self.create_subscription(
            WrenchStamped, 'l_foot/force', self.left_force_callback, 10)
        self.force_sub = self.create_subscription(
            WrenchStamped, 'r_foot/force', self.right_force_callback, 10)

        # Service for high-level commands
        self.walk_srv = self.create_service(
            Trigger, 'start_walking', self.start_walking_callback)
        self.stand_srv = self.create_service(
            Trigger, 'stand_up', self.stand_up_callback)

        # Main control loop
        self.control_timer = self.create_timer(0.01, self.control_loop)  # 100Hz

        # Robot state
        self.joint_states = JointState()
        self.imu_data = Imu()
        self.left_foot_force = WrenchStamped()
        self.right_foot_force = WrenchStamped()
        self.current_behavior = 'standing'

    def joint_state_callback(self, msg):
        self.joint_states = msg

    def imu_callback(self, msg):
        self.imu_data = msg

    def left_force_callback(self, msg):
        self.left_foot_force = msg

    def right_force_callback(self, msg):
        self.right_foot_force = msg

    def control_loop(self):
        """Main control loop running at 100Hz"""
        # Update sensor data
        sensor_data = self.collect_sensor_data()

        # Run behavior-specific controllers
        if self.current_behavior == 'walking':
            motor_commands = self.walking_controller.update(sensor_data)
        elif self.current_behavior == 'balancing':
            motor_commands = self.balance_controller.update(sensor_data)
        elif self.current_behavior == 'standing':
            motor_commands = self.balance_controller.stand_still(sensor_data)
        else:
            # Default: send zero commands
            motor_commands = self.get_zero_commands()

        # Publish motor commands
        self.motor_cmd_pub.publish(motor_commands)

    def collect_sensor_data(self):
        """Collect and process sensor data"""
        sensor_data = SensorData()
        sensor_data.joint_states = self.joint_states
        sensor_data.imu = self.imu_data
        sensor_data.left_foot_force = self.left_foot_force
        sensor_data.right_foot_force = self.right_foot_force
        return sensor_data

    def start_walking_callback(self, request, response):
        self.current_behavior = 'walking'
        response.success = True
        return response

    def stand_up_callback(self, request, response):
        self.current_behavior = 'standing'
        response.success = True
        return response

    def get_zero_commands(self):
        """Return zero motor commands"""
        cmd = MotorCommand()
        cmd.joint_names = self.joint_states.name
        cmd.positions = [0.0] * len(self.joint_states.position)
        cmd.velocities = [0.0] * len(self.joint_states.velocity)
        cmd.efforts = [0.0] * len(self.joint_states.effort)
        return cmd
```

### Walking Controller Implementation

```python
class WalkingController:
    def __init__(self, node):
        self.node = node
        self.foot_step_planner = FootStepPlanner()
        self.trajectory_generator = TrajectoryGenerator()
        self.step_index = 0
        self.current_phase = 'stance'

    def update(self, sensor_data):
        """Update walking controller based on sensor data"""
        # Plan next footsteps
        next_foot_steps = self.foot_step_planner.plan_next_steps(
            sensor_data, self.step_index)

        # Generate trajectories for next phase
        trajectory = self.trajectory_generator.generate_trajectory(
            next_foot_steps, sensor_data, self.step_index)

        # Convert trajectory to motor commands
        motor_cmd = self.trajectory_to_commands(trajectory, sensor_data)

        # Update step index and phase
        self.update_step_phase(trajectory)

        return motor_cmd

    def trajectory_to_commands(self, trajectory, sensor_data):
        """Convert trajectory to motor commands"""
        cmd = MotorCommand()
        # Implementation details for converting desired trajectory
        # to actual motor positions/velocities/efforts
        return cmd

    def update_step_phase(self, trajectory):
        """Update current walking phase (stance, swing, etc.)"""
        # Implementation details for phase management
        pass
```

## Case Study 3: Multi-Modal Perception System

Robotic perception systems often integrate multiple sensors with different data rates and requirements.

### Perception Fusion Node

```python
from sensor_msgs.msg import Image, PointCloud2, LaserScan, CameraInfo
from geometry_msgs.msg import PoseStamped
from my_perception_msgs.msg import FusedPerception, ObjectDetectionArray

class PerceptionFusionNode(Node):
    def __init__(self):
        super().__init__('perception_fusion')

        # Publishers
        self.fused_pub = self.create_publisher(FusedPerception, 'fused_perception', 10)
        self.detection_pub = self.create_publisher(ObjectDetectionArray, 'object_detections', 10)

        # Subscribers with different QoS profiles
        # High-frequency sensors (lidar, IMU)
        self.lidar_sub = self.create_subscription(
            LaserScan, 'scan', self.lidar_callback,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))

        # Medium-frequency sensors (camera)
        self.camera_sub = self.create_subscription(
            Image, 'camera/image_raw', self.camera_callback,
            QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE))

        # Low-frequency sensors (GPS, odometry)
        self.odom_sub = self.create_subscription(
            Odometry, 'odom', self.odom_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE))

        # Synchronization utilities
        from message_filters import ApproximateTimeSynchronizer, Subscriber

        # For sensors that need to be synchronized
        self.image_sub_sync = Subscriber(self, Image, 'camera/image_raw')
        self.camera_info_sub_sync = Subscriber(self, CameraInfo, 'camera/camera_info')

        self.sync = ApproximateTimeSynchronizer(
            [self.image_sub_sync, self.camera_info_sub_sync],
            queue_size=10, slop=0.1)
        self.sync.registerCallback(self.synchronized_callback)

        # Timer for fusion processing
        self.fusion_timer = self.create_timer(0.1, self.fusion_callback)  # 10Hz

        # Perception data storage
        self.latest_lidar = None
        self.latest_image = None
        self.latest_odom = None
        self.latest_camera_info = None

    def lidar_callback(self, msg):
        self.latest_lidar = msg

    def camera_callback(self, msg):
        self.latest_image = msg

    def odom_callback(self, msg):
        self.latest_odom = msg

    def synchronized_callback(self, image_msg, camera_info_msg):
        """Process synchronized image and camera info"""
        self.latest_image = image_msg
        self.latest_camera_info = camera_info_msg

    def fusion_callback(self):
        """Perform sensor fusion and publish results"""
        if not all([self.latest_lidar, self.latest_image, self.latest_odom]):
            return

        # Perform perception fusion
        fused_data = self.perform_fusion(
            self.latest_lidar,
            self.latest_image,
            self.latest_odom,
            self.latest_camera_info
        )

        # Publish fused perception
        self.fused_pub.publish(fused_data)

    def perform_fusion(self, lidar_data, image_data, odom_data, camera_info):
        """Main fusion algorithm"""
        fused_result = FusedPerception()

        # 1. Process lidar data for obstacle detection
        lidar_objects = self.process_lidar_obstacles(lidar_data)

        # 2. Process image data for object recognition
        image_objects = self.process_image_objects(image_data)

        # 3. Fuse data in global coordinate frame
        global_objects = self.transform_to_global_frame(
            lidar_objects, image_objects, odom_data, camera_info)

        # 4. Perform final fusion and filtering
        final_objects = self.final_fusion(global_objects)

        fused_result.objects = final_objects
        fused_result.header.stamp = self.get_clock().now().to_msg()
        fused_result.header.frame_id = 'map'

        return fused_result

    def process_lidar_obstacles(self, lidar_data):
        """Process lidar data for obstacle detection"""
        # Implementation of lidar-based obstacle detection
        return []

    def process_image_objects(self, image_data):
        """Process image data for object recognition"""
        # Implementation of computer vision object detection
        return []

    def transform_to_global_frame(self, lidar_objects, image_objects, odom, camera_info):
        """Transform objects to global coordinate frame"""
        # Implementation of coordinate frame transformations
        return lidar_objects + image_objects

    def final_fusion(self, objects):
        """Final fusion and filtering of objects"""
        # Implementation of data association and filtering
        return objects
```

## Case Study 4: ROS 2 in Space Robotics

Space robotics applications have unique requirements for reliability and fault tolerance.

### Fault-Tolerant Communication

```python
class SpaceRobotController(Node):
    def __init__(self):
        super().__init__('space_robot_controller')

        # Multiple redundant communication paths
        self.primary_pub = self.create_publisher(Command, 'primary_cmd', 10)
        self.backup_pub = self.create_publisher(Command, 'backup_cmd', 10)

        # Heartbeat monitoring
        self.heartbeat_pub = self.create_publisher(String, 'heartbeat', 10)
        self.heartbeat_sub = self.create_subscription(
            String, 'ground_station_heartbeat', self.heartbeat_callback, 10)

        # Critical system monitoring
        self.critical_subscribers = []
        self.system_health = {'communication': True, 'power': True, 'motors': True}

        # Timers with different priorities
        self.critical_timer = self.create_timer(0.01, self.critical_control_loop)
        self.normal_timer = self.create_timer(0.1, self.normal_control_loop)

    def critical_control_loop(self):
        """Critical control loop - runs at high frequency"""
        # Monitor system health
        self.check_system_health()

        # If communication is lost, switch to autonomous mode
        if not self.system_health['communication']:
            self.execute_autonomous_behavior()

    def check_system_health(self):
        """Check health of critical systems"""
        # Check if we've received ground station heartbeat recently
        if self.time_since_last_heartbeat() > 5.0:  # 5 seconds
            self.system_health['communication'] = False
            self.get_logger().warn('Communication with ground station lost')

    def execute_autonomous_behavior(self):
        """Execute safe autonomous behavior when communication is lost"""
        # Implement safe behavior, e.g., return to safe position
        safe_cmd = Command()
        safe_cmd.type = 'safe_mode'
        self.primary_pub.publish(safe_cmd)

    def heartbeat_callback(self, msg):
        """Handle ground station heartbeat"""
        self.last_heartbeat_time = self.get_clock().now()
        self.system_health['communication'] = True
```

## Performance Optimization in Real Systems

### Efficient Message Handling

```python
import threading
from collections import deque

class OptimizedPerceptionNode(Node):
    def __init__(self):
        super().__init__('optimized_perception')

        # Use threading for CPU-intensive processing
        self.processing_thread = threading.Thread(target=self.processing_worker)
        self.processing_queue = deque(maxlen=10)  # Limit queue size
        self.processing_lock = threading.Lock()

        # Publishers and subscribers
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, 1)
        self.result_pub = self.create_publisher(DetectionResult, 'detections', 10)

        # Start processing thread
        self.processing_thread.start()

    def image_callback(self, msg):
        """Non-blocking image callback"""
        with self.processing_lock:
            if len(self.processing_queue) < 10:  # Don't overload queue
                self.processing_queue.append(msg)

    def processing_worker(self):
        """Background processing thread"""
        while rclpy.ok():
            with self.processing_lock:
                if self.processing_queue:
                    msg = self.processing_queue.popleft()
                else:
                    msg = None

            if msg is not None:
                # Process image (CPU-intensive operation)
                result = self.process_image(msg)
                self.result_pub.publish(result)

            time.sleep(0.001)  # Small sleep to prevent busy waiting

    def process_image(self, image_msg):
        """CPU-intensive image processing"""
        # Implementation of image processing
        return DetectionResult()
```

## Debugging and Monitoring in Production

### Diagnostic System

```python
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue

class DiagnosticNode(Node):
    def __init__(self):
        super().__init__('diagnostic_node')

        # Publishers for diagnostics
        self.diag_pub = self.create_publisher(DiagnosticArray, '/diagnostics', 1)

        # Timer for diagnostic updates
        self.diag_timer = self.create_timer(1.0, self.publish_diagnostics)

    def publish_diagnostics(self):
        """Publish system diagnostics"""
        diag_array = DiagnosticArray()
        diag_array.header.stamp = self.get_clock().now().to_msg()

        # CPU usage diagnostic
        cpu_diag = DiagnosticStatus()
        cpu_diag.name = 'CPU Usage'
        cpu_diag.level = DiagnosticStatus.OK
        cpu_usage = self.get_cpu_usage()
        cpu_diag.message = f'CPU usage: {cpu_usage:.1f}%'
        cpu_diag.values = [KeyValue(key='usage_percent', value=str(cpu_usage))]

        if cpu_usage > 80.0:
            cpu_diag.level = DiagnosticStatus.WARN
        elif cpu_usage > 90.0:
            cpu_diag.level = DiagnosticStatus.ERROR

        diag_array.status.append(cpu_diag)

        # Memory usage diagnostic
        mem_diag = DiagnosticStatus()
        mem_diag.name = 'Memory Usage'
        mem_diag.level = DiagnosticStatus.OK
        mem_usage = self.get_memory_usage()
        mem_diag.message = f'Memory usage: {mem_usage:.1f}%'
        mem_diag.values = [KeyValue(key='usage_percent', value=str(mem_usage))]

        if mem_usage > 85.0:
            mem_diag.level = DiagnosticStatus.WARN

        diag_array.status.append(mem_diag)

        # Communication diagnostic
        comm_diag = DiagnosticStatus()
        comm_diag.name = 'Communication Status'
        comm_diag.level = DiagnosticStatus.OK
        comm_diag.message = 'All communication links active'
        diag_array.status.append(comm_diag)

        self.diag_pub.publish(diag_array)

    def get_cpu_usage(self):
        """Get current CPU usage"""
        import psutil
        return psutil.cpu_percent(interval=0.1)

    def get_memory_usage(self):
        """Get current memory usage"""
        import psutil
        return psutil.virtual_memory().percent
```

## Best Practices from Real Applications

### 1. Error Handling and Recovery

```python
class RobustNode(Node):
    def __init__(self):
        super().__init__('robust_node')

        # Implement graceful degradation
        self.fallback_modes = ['full_functionality', 'reduced_functionality', 'safe_mode']
        self.current_mode = 'full_functionality'

    def safe_mode_operation(self):
        """Minimal safe operation when errors occur"""
        # Stop all non-essential functions
        # Move to safe position if possible
        # Wait for recovery or human intervention
        pass

    def recover_from_error(self):
        """Attempt to recover from error state"""
        # Try to reinitialize failed components
        # Return to normal operation if successful
        pass
```

### 2. Resource Management

```python
class ResourceAwareNode(Node):
    def __init__(self):
        super().__init__('resource_aware_node')

        # Monitor resource usage
        self.memory_threshold = 0.8  # 80% memory usage threshold
        self.cpu_threshold = 0.85    # 85% CPU usage threshold

    def check_resources(self):
        """Check if system resources are adequate"""
        import psutil

        memory_percent = psutil.virtual_memory().percent / 100.0
        cpu_percent = psutil.cpu_percent() / 100.0

        if memory_percent > self.memory_threshold:
            self.get_logger().warn(f'High memory usage: {memory_percent:.1%}')
            self.reduce_resource_usage()

        if cpu_percent > self.cpu_threshold:
            self.get_logger().warn(f'High CPU usage: {cpu_percent:.1%}')
            self.reduce_processing_frequency()
```

### 3. Configuration Management

```python
class ConfigurableNode(Node):
    def __init__(self):
        super().__init__('configurable_node')

        # Load configuration from file
        config_path = self.declare_parameter('config_file', '').value
        if config_path:
            self.load_configuration(config_path)

    def load_configuration(self, config_path):
        """Load configuration from YAML file"""
        import yaml
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                self.apply_configuration(config)
        except Exception as e:
            self.get_logger().error(f'Failed to load configuration: {e}')

    def apply_configuration(self, config):
        """Apply loaded configuration"""
        # Update parameters based on configuration
        for param_name, param_value in config.get('parameters', {}).items():
            self.set_parameters([Parameter(param_name, value=param_value)])
```

## Key Takeaways

- Real-world systems require robust error handling and fallback mechanisms
- Performance optimization is crucial for real-time applications
- Resource management prevents system overload
- Diagnostic systems enable proactive maintenance
- Configuration management allows for flexible deployment
- Communication patterns must scale with system complexity
- Safety and reliability are paramount in deployed systems

## Module Summary

In this module, we've covered:

1. **Fundamentals**: Introduction to robotic nervous systems and ROS 2
2. **Architecture**: Understanding ROS 2 concepts and design principles
3. **Primitives**: Nodes and communication primitives
4. **Patterns**: Topics, services, and actions
5. **Configuration**: Parameters and dynamic reconfiguration
6. **Advanced**: Complex communication patterns and optimization
7. **Applications**: Real-world implementations and best practices

These concepts form the foundation of effective robotic communication systems. As you move to the next module on Digital Twins, you'll see how these communication patterns integrate with simulation and modeling systems to create comprehensive robotic solutions.

The Robotic Nervous System (ROS 2) module has provided you with the essential knowledge to design, implement, and maintain robust communication architectures for robotic systems. Whether you're building a simple mobile robot or a complex humanoid system, these principles will guide you in creating effective and reliable communication networks.