---
sidebar_position: 3
title: Nodes and Communication Primitives
---

# Nodes and Communication Primitives

In this chapter, we'll explore the fundamental building blocks of ROS 2: nodes and the communication primitives that enable them to interact. Understanding these concepts is essential for creating interconnected robotic systems.

## Understanding Nodes

A node is the fundamental building block of a ROS 2 system. It's an executable process that performs specific computations and communicates with other nodes. Think of nodes as the "organs" of your robot's nervous system.

### Node Characteristics

- **Independence**: Each node runs as a separate process
- **Specialization**: Each node typically performs a specific function
- **Communication**: Nodes communicate through ROS 2's communication primitives
- **Identification**: Each node has a unique name within the system

### Creating a Node

Let's examine the structure of a basic ROS 2 node:

```cpp
// C++ example of a minimal node
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

class MinimalPublisher : public rclcpp::Node
{
public:
  MinimalPublisher()
  : Node("minimal_publisher"), count_(0)
  {
    publisher_ = this->create_publisher<std_msgs::msg::String>("topic", 10);
    timer_ = this->create_wall_timer(
      500ms, std::bind(&MinimalPublisher::timer_callback, this));
  }

private:
  void timer_callback()
  {
    auto message = std_msgs::msg::String();
    message.data = "Hello, world! " + std::to_string(count_++);
    RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());
    publisher_->publish(message);
  }
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
  size_t count_;
};
```

```python
# Python equivalent
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher = self.create_publisher(String, 'topic', 10)
        self.timer = self.create_timer(0.5, self.timer_callback)
        self.count = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello, world! {self.count}'
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.publisher.publish(msg)
        self.count += 1
```

## Communication Primitives

ROS 2 provides four main communication primitives that enable nodes to interact:

### 1. Topics (Publish/Subscribe)

Topics use a publish/subscribe model for asynchronous, one-to-many communication:

```python
# Publisher example
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

class ImagePublisher(Node):
    def __init__(self):
        super().__init__('image_publisher')
        self.publisher = self.create_publisher(Image, 'camera/image_raw', 10)

    def publish_image(self, image_data):
        msg = Image()
        # Fill in image data...
        self.publisher.publish(msg)

# Subscriber example
class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.subscription = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            10)

    def image_callback(self, msg):
        self.get_logger().info('Received image data')
        # Process image...
```

### 2. Services (Request/Response)

Services provide synchronous, one-to-one communication:

```python
# Service server
from example_interfaces.srv import AddTwoInts

class AddTwoIntsServer(Node):
    def __init__(self):
        super().__init__('add_two_ints_server')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_callback)

    def add_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Returning: {response.sum}')
        return response

# Service client
class AddTwoIntsClient(Node):
    def __init__(self):
        super().__init__('add_two_ints_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')

    def send_request(self, a, b):
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        request = AddTwoInts.Request()
        request.a = a
        request.b = b
        self.future = self.cli.call_async(request)
        return self.future
```

### 3. Actions (Goal/Feedback/Result)

Actions handle long-running tasks with progress feedback:

```python
from rclpy.action import ActionClient
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose

class NavigateActionClient(Node):
    def __init__(self):
        super().__init__('navigate_action_client')
        self._action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

    def send_goal(self, pose):
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose

        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback)

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Received feedback: {feedback}')
```

### 4. Parameters

Parameters provide a way to configure nodes at runtime:

```python
class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')

        # Declare parameters with default values
        self.declare_parameter('robot_name', 'my_robot')
        self.declare_parameter('max_velocity', 1.0)
        self.declare_parameter('safety_distance', 0.5)

        # Get parameter values
        self.robot_name = self.get_parameter('robot_name').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.safety_distance = self.get_parameter('safety_distance').value

        # Set up parameter callback for dynamic reconfiguration
        self.add_on_set_parameters_callback(self.parameter_callback)

    def parameter_callback(self, params):
        for param in params:
            if param.name == 'max_velocity' and param.type_ == Parameter.Type.PARAMETER_DOUBLE:
                self.get_logger().info(f'Updated max_velocity to: {param.value}')
        return SetParametersResult(successful=True)
```

## Node Lifecycle

ROS 2 nodes can have a lifecycle that provides better control over their state:

```python
from lifecycle_msgs.msg import Transition
from lifecycle_msgs.srv import ChangeState
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn

class LifecycleExample(LifecycleNode):
    def __init__(self):
        super().__init__('lifecycle_example')

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f'Configuring node from state {state.label}')
        # Initialize resources
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f'Activating node from state {state.label}')
        # Start operations
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f'Deactivating node from state {state.label}')
        # Pause operations
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f'Cleaning up node from state {state.label}')
        # Release resources
        return TransitionCallbackReturn.SUCCESS
```

## Best Practices for Node Design

### 1. Single Responsibility Principle
Each node should have a clear, single purpose:

```python
# Good: Dedicated sensor node
class LidarSensorNode(Node):
    def __init__(self):
        super().__init__('lidar_sensor')
        # Only handles lidar sensor operations

# Good: Dedicated controller node
class MotionControllerNode(Node):
    def __init__(self):
        super().__init__('motion_controller')
        # Only handles motion control
```

### 2. Proper Error Handling
Nodes should handle errors gracefully:

```python
def sensor_callback(self, msg):
    try:
        processed_data = self.process_sensor_data(msg)
        self.publisher.publish(processed_data)
    except Exception as e:
        self.get_logger().error(f'Error processing sensor data: {e}')
        # Continue operation or implement recovery strategy
```

### 3. Resource Management
Properly manage resources and clean up when needed:

```python
def destroy_node(self):
    # Clean up resources before destroying the node
    if hasattr(self, 'timer'):
        self.timer.cancel()
    if hasattr(self, 'publisher'):
        self.publisher.destroy()
    if hasattr(self, 'subscription'):
        self.subscription.destroy()
    super().destroy_node()
```

## Practical Example: Simple Robot Controller

Let's build a practical example combining multiple concepts:

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')

        # Publisher for velocity commands
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # Subscriber for sensor data
        self.laser_sub = self.create_subscription(
            LaserScan, 'scan', self.laser_callback, 10)

        # Publisher for safety status
        self.safety_pub = self.create_publisher(Bool, 'safety_status', 10)

        # Timer for control loop
        self.control_timer = self.create_timer(0.1, self.control_loop)

        # Robot state
        self.obstacle_detected = False
        self.safety_enabled = True

        # Parameters
        self.declare_parameter('safety_distance', 0.5)
        self.declare_parameter('max_linear_velocity', 0.5)
        self.declare_parameter('max_angular_velocity', 1.0)

    def laser_callback(self, msg):
        # Check for obstacles in front of the robot
        min_distance = min(msg.ranges)
        safety_distance = self.get_parameter('safety_distance').value

        self.obstacle_detected = min_distance < safety_distance

        # Publish safety status
        safety_msg = Bool()
        safety_msg.data = not self.obstacle_detected
        self.safety_pub.publish(safety_msg)

    def control_loop(self):
        if not self.safety_enabled:
            # Emergency stop
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_vel_pub.publish(cmd)
            return

        cmd = Twist()
        if self.obstacle_detected:
            # Stop when obstacle detected
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        else:
            # Move forward
            max_vel = self.get_parameter('max_linear_velocity').value
            cmd.linear.x = max_vel
            cmd.angular.z = 0.0

        self.cmd_vel_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    controller = RobotController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Key Takeaways

- Nodes are the fundamental building blocks of ROS 2 systems
- Four main communication primitives: topics, services, actions, and parameters
- Each primitive serves different communication patterns and use cases
- Proper node design follows single responsibility and error handling principles
- The robot controller example demonstrates combining multiple concepts

## Next Steps

In the next chapter, we'll explore topics, services, and actions in greater detail, including advanced usage patterns and Quality of Service configurations.