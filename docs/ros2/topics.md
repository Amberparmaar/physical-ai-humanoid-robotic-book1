---
sidebar_label: ROS2 Topics
---

# ROS2 Topics

## What are Topics?

Topics in ROS2 provide a publish-subscribe communication pattern that enables asynchronous communication between nodes. Publishers send messages to topics, and subscribers receive messages from topics without direct knowledge of each other.

## Topic Characteristics

- **Asynchronous**: Publishers and subscribers operate independently
- **Anonymous**: Nodes are not aware of each other's identities
- **Many-to-many**: Multiple publishers can publish to a topic, and multiple subscribers can subscribe to a topic
- **Message-based**: All communication happens through serialized messages

## Quality of Service (QoS)

ROS2 introduces Quality of Service settings that allow fine-tuning of topic behavior:

### Reliability Policy
- **Reliable**: All messages are guaranteed to be delivered
- **Best Effort**: Messages may be lost but delivery is faster

### Durability Policy
- **Transient Local**: Publishers send last message to new subscribers
- **Volatile**: New subscribers only receive new messages

### History Policy
- **Keep Last**: Store the last N messages
- **Keep All**: Store all messages (memory intensive)

### Rate Settings
- **Depth**: Number of messages to store in publisher/subscriber queues

## Example Topic Implementation

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    minimal_subscriber = MinimalSubscriber()
    
    rclpy.spin(minimal_publisher)
    rclpy.spin(minimal_subscriber)

    minimal_publisher.destroy_node()
    minimal_subscriber.destroy_node()
    rclpy.shutdown()
```

## Topic Design Considerations

### Bandwidth Usage
- High-frequency topics can consume significant network bandwidth
- Consider message size and publishing frequency
- Use compression for large data like images or point clouds

### Synchronization
- Topics don't guarantee message synchronization
- Use message filters for time synchronization between multiple topics
- Consider using services or actions for synchronized communication

### Performance
- Choose appropriate QoS settings for your application
- Use intra-process communication when nodes are in the same process
- Monitor topic statistics for performance debugging

## Topics in Humanoid Robotics

Humanoid robots use topics extensively for:

- **Sensor Data**: IMU, joint states, camera feeds, LIDAR scans
- **Actuator Commands**: Joint position/velocity/effort commands
- **State Information**: Robot pose, joint states, battery levels
- **Perception Results**: Object detection, SLAM results, person tracking
- **Behavior Outputs**: Intention messages, state changes

## Debugging Topics

Useful tools for topic debugging:

- `ros2 topic list`: List all available topics
- `ros2 topic echo <topic_name>`: Print messages from a topic
- `ros2 topic info <topic_name>`: Show information about a topic
- `ros2 topic pub <topic_name> <msg_type> <args>`: Publish to a topic from command line

Topics form the backbone of communication in humanoid robotics systems, enabling the distributed architecture necessary for complex robot control.