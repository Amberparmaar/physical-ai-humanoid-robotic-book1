---
sidebar_label: ROS2 Nodes
---

# ROS2 Nodes

## Understanding Nodes

A node in ROS2 is an executable process that works in concert with other nodes to perform computational work. Nodes are the fundamental building blocks of a ROS2 application and can publish to or subscribe to topics, provide services, or execute actions.

## Creating a Node

A basic ROS2 node requires the following components:

1. **Initialization**: Initialize the ROS2 client library
2. **Node creation**: Create a node instance with a unique name
3. **Spin**: Keep the node running and processing callbacks
4. **Cleanup**: Shutdown the node properly

## Node Communication Patterns

### Publisher/Subscriber (Topics)
Nodes can communicate asynchronously using topics. Publishers send messages to topics, and subscribers receive messages from topics.

```python
# Publisher example
publisher = node.create_publisher(String, 'topic_name', 10)

# Subscriber example
subscriber = node.create_subscription(String, 'topic_name', callback, 10)
```

### Client/Service
Nodes can communicate synchronously using services. Clients send requests to services, which respond with results.

```python
# Service server example
service = node.create_service(AddTwoInts, 'add_two_ints', handle_add_two_ints)

# Client example
client = node.create_client(AddTwoInts, 'add_two_ints')
```

### Action Client/Server
For long-running tasks with feedback, nodes use actions.

```python
# Action server example
action_server = ActionServer(
    node,
    Fibonacci,
    'fibonacci',
    execute_callback=execute_callback
)

# Action client example
action_client = ActionClient(node, Fibonacci, 'fibonacci')
```

## Node Lifecycle

ROS2 nodes follow a lifecycle management system that allows for more sophisticated state management:

- **Unconfigured**: Node is created but not configured
- **Inactive**: Node is configured but not active
- **Active**: Node is running normally
- **Finalized**: Node is shutdown and cannot be reactivated

## Best Practices for Node Design

- Keep nodes focused on a single responsibility
- Use parameters for configuration rather than hardcoding values
- Implement proper error handling and logging
- Use composition where possible instead of multiple separate nodes
- Implement node lifecycle management when appropriate

## Nodes in Humanoid Robotics

In humanoid robotics, nodes typically handle specialized functions such as:

- Sensor data processing (IMU, cameras, force/torque sensors)
- Motor control interfaces
- Motion planning algorithms
- Perception systems
- Behavior managers
- Communication with external systems