---
sidebar_position: 4
title: Topics, Services, and Actions
---

# Topics, Services, and Actions

In this chapter, we'll explore the three primary communication patterns in ROS 2: topics (publish/subscribe), services (request/response), and actions (goal-oriented communication). Understanding these patterns is crucial for designing effective robotic communication architectures.

## Topics: Publish/Subscribe Pattern

Topics implement an asynchronous, one-to-many communication pattern. Publishers send messages to topics, and any number of subscribers can receive those messages.

### Topic Characteristics

- **Asynchronous**: Publishers don't wait for responses
- **One-to-many**: Multiple subscribers can receive the same message
- **Data-driven**: Communication is triggered by data availability
- **Fire-and-forget**: Publishers don't know if messages are received

### Creating Topics

```python
# Publisher
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Talker(Node):
    def __init__(self):
        super().__init__('talker')
        self.publisher = self.create_publisher(String, 'chatter', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

# Subscriber
class Listener(Node):
    def __init__(self):
        super().__init__('listener')
        self.subscription = self.create_subscription(
            String,
            'chatter',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')
```

### Quality of Service (QoS) for Topics

QoS policies allow fine-tuning of topic behavior:

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

# Create a QoS profile for reliable communication
reliable_qos = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE
)

# Create a QoS profile for best-effort communication
best_effort_qos = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE
)

# Use different QoS for different data types
class SensorNode(Node):
    def __init__(self):
        super().__init__('sensor_node')

        # Reliable for critical data
        self.critical_pub = self.create_publisher(
            String, 'critical_data', reliable_qos)

        # Best effort for high-frequency data
        self.video_pub = self.create_publisher(
            Image, 'video_stream', best_effort_qos)
```

### Common Topic Use Cases

- **Sensor data**: Camera images, lidar scans, IMU data
- **Robot state**: Joint positions, odometry, battery status
- **Control commands**: Velocity commands, actuator positions
- **Event notifications**: System status, warnings, alarms

## Services: Request/Response Pattern

Services provide synchronous, one-to-one communication where a client sends a request and waits for a response.

### Service Characteristics

- **Synchronous**: Client waits for response
- **One-to-one**: One client communicates with one server
- **Request-response**: Defined request and response message types
- **Blocking**: Client is blocked until response is received

### Creating Services

First, define a service interface (e.g., `AddTwoInts.srv`):
```
int64 a
int64 b
---
int64 sum
```

Then implement the server and client:

```python
# Service Server
from example_interfaces.srv import AddTwoInts

class AddTwoIntsServer(Node):
    def __init__(self):
        super().__init__('add_two_ints_server')
        self.srv = self.create_service(
            AddTwoInts,
            'add_two_ints',
            self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Returning {response.sum}')
        return response

# Service Client
class AddTwoIntsClient(Node):
    def __init__(self):
        super().__init__('add_two_ints_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')

    def send_request(self, a, b):
        # Wait for service to be available
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        # Create and send request
        request = AddTwoInts.Request()
        request.a = a
        request.b = b

        # Send request asynchronously
        self.future = self.cli.call_async(request)
        return self.future

def main(args=None):
    rclpy.init(args=args)

    client = AddTwoIntsClient()
    future = client.send_request(1, 2)

    # Spin until the response is received
    rclpy.spin_until_future_complete(client, future)

    if future.result() is not None:
        response = future.result()
        client.get_logger().info(f'Result: {response.sum}')
    else:
        client.get_logger().info('Service call failed')

    client.destroy_node()
    rclpy.shutdown()
```

### Common Service Use Cases

- **Configuration**: Setting parameters, loading maps
- **Short operations**: Calibration, diagnostics
- **State queries**: Getting robot status, sensor readings
- **One-time commands**: Taking photos, saving data

## Actions: Goal-Oriented Communication

Actions are designed for long-running tasks that require feedback and goal management. They combine aspects of both topics and services.

### Action Characteristics

- **Long-running**: For operations that take significant time
- **Goal-oriented**: Client sends a goal, server reports progress
- **Feedback**: Continuous updates on operation progress
- **Cancelation**: Clients can cancel goals in progress

### Creating Actions

First, define an action interface (e.g., `Fibonacci.action`):
```
int32 order
---
int32[] sequence
---
int32[] partial_sequence
```

Then implement the action server and client:

```python
# Action Server
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class FibonacciActionServer(Node):
    def __init__(self):
        super().__init__('fibonacci_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback)

    def goal_callback(self, goal_request):
        self.get_logger().info('Received goal request')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')

        feedback_msg = Fibonacci.Feedback()
        feedback_msg.partial_sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Fibonacci.Result()

            feedback_msg.partial_sequence.append(
                feedback_msg.partial_sequence[i] + feedback_msg.partial_sequence[i-1])

            goal_handle.publish_feedback(feedback_msg)
            time.sleep(1)  # Simulate work

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.partial_sequence
        self.get_logger().info(f'Result: {result.sequence}')
        return result

# Action Client
from rclpy.action import ActionClient
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class FibonacciActionClient(Node):
    def __init__(self):
        super().__init__('fibonacci_action_client')
        self._action_client = ActionClient(self, Fibonacci, 'fibonacci')

    def send_goal(self, order):
        goal_msg = Fibonacci.Goal()
        goal_msg.order = order

        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback)

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        self.get_logger().info(f'Received feedback: {feedback_msg.partial_sequence}')

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: {result.sequence}')
```

### Common Action Use Cases

- **Navigation**: Moving to a goal location with progress feedback
- **Manipulation**: Moving robot arms with trajectory feedback
- **Data processing**: Long-running analysis tasks
- **Calibration**: Multi-step calibration procedures

## Choosing the Right Communication Pattern

### When to Use Topics

- **Data streaming**: Camera feeds, sensor data
- **Event broadcasting**: System status, alarms
- **Control commands**: Velocity, position commands
- **When you don't need acknowledgment**: Fire-and-forget communication

### When to Use Services

- **One-time operations**: Taking a photo, getting current position
- **Queries**: Requesting robot status, parameters
- **Configuration**: Setting parameters, loading maps
- **When you need a response**: Synchronous request-response pattern

### When to Use Actions

- **Long-running operations**: Navigation, manipulation
- **When you need progress feedback**: Task progress updates
- **Cancelable operations**: Ability to stop in-progress tasks
- **Complex workflows**: Multi-step operations with state

## Advanced Communication Patterns

### Publisher with Custom QoS

```python
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy

class AdvancedPublisher(Node):
    def __init__(self):
        super().__init__('advanced_publisher')

        # Custom QoS for different scenarios
        # For critical data that must be delivered
        critical_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        # For high-frequency data where some loss is acceptable
        stream_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        self.critical_publisher = self.create_publisher(String, 'critical_topic', critical_qos)
        self.stream_publisher = self.create_publisher(Image, 'stream_topic', stream_qos)
```

### Multiple Subscriptions

```python
class MultiSubscriber(Node):
    def __init__(self):
        super().__init__('multi_subscriber')

        # Subscribe to multiple topics
        self.odom_subscription = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10)
        self.imu_subscription = self.create_subscription(
            Imu, 'imu', self.imu_callback, 10)
        self.laser_subscription = self.create_subscription(
            LaserScan, 'scan', self.laser_callback, 10)

        # Store data from different sensors
        self.odom_data = None
        self.imu_data = None
        self.laser_data = None

    def odom_callback(self, msg):
        self.odom_data = msg
        self.process_fusion()

    def imu_callback(self, msg):
        self.imu_data = msg
        self.process_fusion()

    def laser_callback(self, msg):
        self.laser_data = msg
        self.process_fusion()

    def process_fusion(self):
        # Process data from multiple sensors
        if self.odom_data and self.imu_data:
            # Perform sensor fusion
            pass
```

## Practical Example: Robot Navigation System

Let's build a practical example combining all three communication patterns:

```python
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile

from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from example_interfaces.srv import Trigger
from example_interfaces.action import NavigateToPose

class NavigationManager(Node):
    def __init__(self):
        super().__init__('navigation_manager')

        # Publishers for velocity commands
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # Subscribers for sensor data
        self.odom_sub = self.create_subscription(Odometry, 'odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)

        # Service client for emergency stop
        self.emergency_stop_client = self.create_client(Trigger, 'emergency_stop')

        # Action client for navigation
        self.nav_action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Robot state
        self.current_pose = None
        self.safety_distance = 0.5
        self.obstacle_detected = False

        self.get_logger().info('Navigation Manager initialized')

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose

    def scan_callback(self, msg):
        # Check for obstacles
        if msg.ranges:
            min_range = min(msg.ranges)
            self.obstacle_detected = min_range < self.safety_distance

    def navigate_to_goal(self, x, y, theta):
        """Send navigation goal using action"""
        if not self.nav_action_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().error('Navigation action server not available')
            return False

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.orientation.z = theta  # Simplified orientation

        self.nav_action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.navigation_feedback)

        return True

    def navigation_feedback(self, feedback_msg):
        self.get_logger().info(f'Navigation progress: {feedback_msg.feedback.distance_remaining:.2f}m remaining')

    def emergency_stop(self):
        """Call emergency stop service"""
        if not self.emergency_stop_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('Emergency stop service not available')
            return False

        request = Trigger.Request()
        future = self.emergency_stop_client.call_async(request)
        return future

def main(args=None):
    rclpy.init(args=args)
    nav_manager = NavigationManager()

    try:
        # Example: Navigate to a goal
        nav_manager.navigate_to_goal(1.0, 2.0, 0.0)
        rclpy.spin(nav_manager)
    except KeyboardInterrupt:
        pass
    finally:
        nav_manager.destroy_node()
        rclpy.shutdown()
```

## Key Takeaways

- Topics: Asynchronous, one-to-many communication for data streaming
- Services: Synchronous, one-to-one communication for request-response
- Actions: Goal-oriented communication with feedback for long-running tasks
- QoS policies allow fine-tuning of communication behavior
- Choose the right pattern based on your communication requirements
- Each pattern serves different use cases in robotic systems

## Next Steps

In the next chapter, we'll explore parameters and configuration in ROS 2, including dynamic reconfiguration and parameter management strategies.