---
sidebar_position: 2
title: ROS 2 Architecture and Concepts
---

# ROS 2 Architecture and Concepts

In this chapter, we'll dive deep into the architecture of ROS 2, exploring its core concepts and design principles. Understanding ROS 2 architecture is crucial for building robust robotic systems.

## What is ROS 2?

Robot Operating System 2 (ROS 2) is not an operating system in the traditional sense, but rather a flexible framework for writing robotic software. It provides libraries, tools, and conventions that help developers create complex robotic behaviors by composing simple, reusable components.

## Key Design Principles

### 1. Distributed Computing
ROS 2 is built from the ground up to support distributed computing. This means that different parts of your robot's software can run on different computers, processors, or even in the cloud, all communicating seamlessly.

### 2. Language Independence
Unlike many frameworks that lock you into a single programming language, ROS 2 supports multiple languages including C++, Python, Java, and more. This allows teams to use the best language for each component.

### 3. Real-time Support
ROS 2 includes support for real-time systems, which is crucial for many robotic applications where timing is critical for safety and performance.

### 4. Quality of Service (QoS)
ROS 2 provides Quality of Service policies that allow you to specify requirements for communication between nodes, such as reliability, durability, and performance characteristics.

## Core Architecture Components

### Nodes
A node is a process that performs computation. In ROS 2, nodes are the basic execution units of a ROS program. Each node runs independently and communicates with other nodes through messages.

```python
# Example of a simple ROS 2 node in Python
import rclpy
from rclpy.node import Node

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello, world!'
        self.publisher.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
```

### Topics and Messages
Topics are named buses over which nodes exchange messages. Messages are the data packets sent between nodes. ROS 2 uses a publish-subscribe communication model where publishers send messages to topics and subscribers receive messages from topics.

### Services
Services provide a request-response communication pattern. A client sends a request to a service and waits for a response. This is useful for operations that require a specific response.

### Actions
Actions are a more advanced communication pattern that supports long-running tasks with feedback and goal management. They're ideal for operations like navigation where you need to track progress.

## DDS: The Underlying Middleware

ROS 2 uses Data Distribution Service (DDS) as its underlying communication middleware. DDS is a standard for distributed, real-time systems that provides:

- **Discovery**: Automatic discovery of nodes on the network
- **Reliability**: Guaranteed delivery of messages
- **Real-time performance**: Low latency communication
- **Quality of Service**: Configurable communication policies

## Client Libraries

ROS 2 provides client libraries (rcl) that implement the ROS 2 API for different programming languages:

- **rclcpp**: For C++ applications
- **rclpy**: For Python applications
- **rcljava**: For Java applications
- **rclnodejs**: For Node.js applications

## Package Management

ROS 2 uses ament as its build system and package manager. Packages contain:

- Source code
- Configuration files
- Dependencies
- Documentation
- Tests

## Communication Patterns

### Publisher-Subscriber (Topics)
- Asynchronous communication
- One-to-many or many-to-many
- Data flow: publisher → topic → subscriber(s)

### Client-Server (Services)
- Synchronous communication
- One-to-one
- Request-response pattern

### Action Server-Client
- Asynchronous with feedback
- One-to-one
- Goal-oriented with progress tracking

## Practical Example: Sensor Network

Let's consider a practical example of a robot with multiple sensors:

```python
# Sensor node example
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PointStamped

class SensorNetworkNode(Node):
    def __init__(self):
        super().__init__('sensor_network')

        # Publisher for processed sensor data
        self.publisher = self.create_publisher(PointStamped, 'processed_lidar_data', 10)

        # Subscriber for raw lidar data
        self.subscription = self.create_subscription(
            LaserScan,
            'raw_lidar',
            self.lidar_callback,
            10)

        self.get_logger().info('Sensor network node initialized')

    def lidar_callback(self, msg):
        # Process lidar data
        processed_data = self.process_lidar(msg)

        # Publish processed data
        self.publisher.publish(processed_data)

    def process_lidar(self, scan_msg):
        # Process the raw lidar scan
        # Return processed data as PointStamped
        point = PointStamped()
        point.header = scan_msg.header
        # Processing logic here...
        return point
```

## Key Takeaways

- ROS 2 provides a distributed computing framework for robotics
- The architecture is based on nodes, topics, services, and actions
- DDS serves as the underlying communication middleware
- Quality of Service policies allow fine-tuning of communication behavior
- The system supports multiple programming languages
- Packages provide a modular way to organize code

## Next Steps

In the next chapter, we'll explore nodes and communication primitives in more detail, including hands-on examples of creating and running ROS 2 nodes.