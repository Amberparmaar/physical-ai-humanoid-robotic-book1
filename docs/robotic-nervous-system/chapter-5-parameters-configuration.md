---
sidebar_position: 5
title: Parameters and Configuration
---

# Parameters and Configuration

In this chapter, we'll explore ROS 2's parameter system, which provides a powerful way to configure nodes at runtime. Parameters enable dynamic reconfiguration of robotic systems without requiring code changes or restarts.

## Understanding Parameters

Parameters in ROS 2 provide a way to configure node behavior at runtime. They serve as the "settings" or "knobs" that can be adjusted to modify how a node operates without changing its source code.

### Parameter Characteristics

- **Dynamic**: Can be changed at runtime
- **Typed**: Support various data types (int, float, string, bool, lists)
- **Hierarchical**: Organized in a namespace structure
- **Accessible**: Can be accessed via command line tools or API
- **Persistent**: Can be saved and loaded from files

### Parameter Types

ROS 2 supports the following parameter types:

- **Integer**: `int`, `int8`, `int16`, `int32`, `int64`
- **Unsigned Integer**: `uint8`, `uint16`, `uint32`, `uint64`
- **Floating Point**: `float`, `double`
- **Boolean**: `bool`
- **String**: `string`
- **Arrays**: Lists of any of the above types
- **Byte Array**: `byte_array`

## Declaring and Using Parameters

### Basic Parameter Declaration

```python
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')

        # Declare parameters with default values
        self.declare_parameter('robot_name', 'my_robot')
        self.declare_parameter('max_velocity', 1.0)
        self.declare_parameter('safety_enabled', True)
        self.declare_parameter('wheel_diameters', [0.1, 0.1, 0.1, 0.1])  # For a 4-wheeled robot

        # Get parameter values
        self.robot_name = self.get_parameter('robot_name').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.safety_enabled = self.get_parameter('safety_enabled').value
        self.wheel_diameters = self.get_parameter('wheel_diameters').value

        self.get_logger().info(f'Robot name: {self.robot_name}')
        self.get_logger().info(f'Max velocity: {self.max_velocity}')
        self.get_logger().info(f'Safety enabled: {self.safety_enabled}')
        self.get_logger().info(f'Wheel diameters: {self.wheel_diameters}')
```

### Parameter with Descriptions and Constraints

```python
from rcl_interfaces.msg import ParameterDescriptor, IntegerRange, FloatingPointRange

class ConstrainedParameterNode(Node):
    def __init__(self):
        super().__init__('constrained_parameter_node')

        # Declare parameter with description and constraints
        velocity_descriptor = ParameterDescriptor()
        velocity_descriptor.description = 'Maximum linear velocity of the robot'
        velocity_descriptor.floating_point_range = [
            FloatingPointRange(from_value=0.0, to_value=5.0, step=0.1)
        ]

        safety_descriptor = ParameterDescriptor()
        safety_descriptor.description = 'Whether safety checks are enabled'
        safety_descriptor.read_only = False  # Parameter can be modified

        # Declare parameters with descriptors
        self.declare_parameter(
            'max_linear_velocity',
            1.0,
            descriptor=velocity_descriptor
        )

        self.declare_parameter(
            'safety_enabled',
            True,
            descriptor=safety_descriptor
        )
```

## Dynamic Parameter Callbacks

Nodes can respond to parameter changes in real-time using callbacks:

```python
from rcl_interfaces.msg import SetParametersResult

class DynamicParameterNode(Node):
    def __init__(self):
        super().__init__('dynamic_parameter_node')

        # Declare parameters
        self.declare_parameter('max_velocity', 1.0)
        self.declare_parameter('safety_distance', 0.5)

        # Get initial values
        self.max_velocity = self.get_parameter('max_velocity').value
        self.safety_distance = self.get_parameter('safety_distance').value

        # Set up parameter callback
        self.add_on_set_parameters_callback(self.parameter_callback)

        self.get_logger().info(f'Initial max velocity: {self.max_velocity}')
        self.get_logger().info(f'Initial safety distance: {self.safety_distance}')

    def parameter_callback(self, params):
        """
        Callback function that is called when parameters are set.
        This function can validate and respond to parameter changes.
        """
        result = SetParametersResult()
        result.successful = True

        for param in params:
            if param.name == 'max_velocity':
                if param.type_ == Parameter.Type.PARAMETER_DOUBLE:
                    if 0.0 <= param.value <= 5.0:  # Validate range
                        self.max_velocity = param.value
                        self.get_logger().info(f'Updated max_velocity to: {param.value}')
                    else:
                        result.successful = False
                        result.reason = 'max_velocity must be between 0.0 and 5.0'
                else:
                    result.successful = False
                    result.reason = 'max_velocity must be a double value'
            elif param.name == 'safety_distance':
                if param.type_ == Parameter.Type.PARAMETER_DOUBLE:
                    if param.value > 0.0:  # Validate positive value
                        self.safety_distance = param.value
                        self.get_logger().info(f'Updated safety_distance to: {param.value}')
                    else:
                        result.successful = False
                        result.reason = 'safety_distance must be positive'
                else:
                    result.successful = False
                    result.reason = 'safety_distance must be a double value'

        return result
```

## Command Line Parameter Operations

ROS 2 provides command-line tools for parameter management:

```bash
# List all parameters of a node
ros2 param list /parameter_node

# Get a specific parameter value
ros2 param get /parameter_node max_velocity

# Set a parameter value
ros2 param set /parameter_node max_velocity 2.5

# Get parameter descriptions
ros2 param describe /parameter_node max_velocity

# Save parameters to a file
ros2 param dump /parameter_node --output config/robot_params.yaml

# Load parameters from a file
ros2 param load /parameter_node config/robot_params.yaml
```

## Parameter Files and Launch Integration

Parameters can be defined in YAML files and loaded at startup:

```yaml
# config/robot_config.yaml
/**:
  ros__parameters:
    robot_name: "my_robot"
    max_velocity: 1.0
    safety_enabled: true
    wheel_diameters: [0.1, 0.1, 0.1, 0.1]
    sensor_offsets:
      - {x: 0.1, y: 0.0, z: 0.0}
      - {x: -0.1, y: 0.0, z: 0.0}
```

Using parameters in launch files:

```python
# launch/robot_launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
import os

def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('my_robot_package'),
        'config',
        'robot_config.yaml'
    )

    robot_node = Node(
        package='my_robot_package',
        executable='robot_node',
        name='robot_node',
        parameters=[config, {'extra_param': 'value'}],
        output='screen'
    )

    return LaunchDescription([robot_node])
```

## Advanced Parameter Patterns

### Parameter Validation with Custom Logic

```python
from rcl_interfaces.msg import SetParametersResult
from rclpy.validate_parameter import parameter_type_from_value

class ValidatedParameterNode(Node):
    def __init__(self):
        super().__init__('validated_parameter_node')

        # Declare parameters
        self.declare_parameter('control_mode', 'velocity')
        self.declare_parameter('pid_gains', [1.0, 0.1, 0.05])  # [P, I, D]
        self.declare_parameter('wheel_radius', 0.05)

        # Set up parameter callback for validation
        self.add_on_set_parameters_callback(self.validate_parameters)

    def validate_parameters(self, params):
        result = SetParametersResult()
        result.successful = True

        for param in params:
            if param.name == 'control_mode':
                # Validate control mode is one of allowed values
                allowed_modes = ['velocity', 'position', 'effort']
                if param.value in allowed_modes:
                    self.get_logger().info(f'Set control mode to: {param.value}')
                else:
                    result.successful = False
                    result.reason = f'control_mode must be one of {allowed_modes}'
                    return result  # Early return on validation failure

            elif param.name == 'pid_gains':
                # Validate PID gains format and values
                if not isinstance(param.value, list) or len(param.value) != 3:
                    result.successful = False
                    result.reason = 'pid_gains must be a list of 3 values [P, I, D]'
                    return result

                # Validate that all values are positive
                if any(gain < 0 for gain in param.value):
                    result.successful = False
                    result.reason = 'PID gains must be non-negative'
                    return result

                self.get_logger().info(f'Set PID gains to: {param.value}')

            elif param.name == 'wheel_radius':
                # Validate wheel radius is positive
                if param.value <= 0:
                    result.successful = False
                    result.reason = 'wheel_radius must be positive'
                    return result

        return result
```

### Parameter Synchronization Across Nodes

```python
class ParameterSyncNode(Node):
    def __init__(self):
        super().__init__('parameter_sync_node')

        # Declare common parameters
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('tf_timeout', 1.0)
        self.declare_parameter('robot_radius', 0.3)

        # Timer to periodically synchronize parameters with other nodes
        self.sync_timer = self.create_timer(10.0, self.synchronize_parameters)

    def synchronize_parameters(self):
        """
        Synchronize parameters with other nodes in the system.
        This is useful for maintaining consistent configuration across the system.
        """
        # Example: Get parameters from other nodes
        try:
            # This would typically involve service calls or other communication
            # to ensure parameter consistency across nodes
            pass
        except Exception as e:
            self.get_logger().error(f'Parameter synchronization failed: {e}')
```

## Parameter Best Practices

### 1. Use Descriptive Names

```python
# Good: Descriptive parameter names
self.declare_parameter('navigation.max_linear_velocity', 1.0)
self.declare_parameter('navigation.min_obstacle_distance', 0.5)
self.declare_parameter('controller.pid_proportional_gain', 1.0)

# Avoid: Generic parameter names
self.declare_parameter('max_vel', 1.0)  # Unclear what max_vel refers to
```

### 2. Provide Default Values

```python
# Always provide sensible default values
self.declare_parameter('safety.timeout', 5.0)  # 5 seconds default
self.declare_parameter('robot.max_speed', 1.0)  # 1 m/s default
self.declare_parameter('debug.enabled', False)  # Debug off by default
```

### 3. Validate Parameter Changes

```python
def validate_parameters(self, params):
    result = SetParametersResult()
    result.successful = True

    for param in params:
        if param.name == 'max_velocity':
            # Validate that max_velocity is within safe bounds
            if not (0.0 <= param.value <= 10.0):
                result.successful = False
                result.reason = f'max_velocity {param.value} is out of safe range [0.0, 10.0]'
                return result

    return result
```

### 4. Document Parameters

```python
def __init__(self):
    super().__init__('documented_parameter_node')

    # Document what each parameter controls
    safety_desc = ParameterDescriptor()
    safety_desc.description = 'Safety timeout in seconds. Robot stops if no valid commands received within this time.'

    self.declare_parameter(
        'safety.timeout',
        5.0,
        descriptor=safety_desc
    )
```

## Practical Example: Adaptive Robot Controller

Let's build a practical example that demonstrates dynamic parameter adjustment:

```python
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import math

class AdaptiveController(Node):
    def __init__(self):
        super().__init__('adaptive_controller')

        # Declare configurable parameters
        self.declare_parameter('linear_velocity', 0.5)
        self.declare_parameter('angular_velocity', 0.5)
        self.declare_parameter('safety_distance', 0.5)
        self.declare_parameter('adaptive_enabled', True)
        self.declare_parameter('obstacle_approach_factor', 2.0)

        # Initialize with current parameter values
        self.linear_velocity = self.get_parameter('linear_velocity').value
        self.angular_velocity = self.get_parameter('angular_velocity').value
        self.safety_distance = self.get_parameter('safety_distance').value
        self.adaptive_enabled = self.get_parameter('adaptive_enabled').value
        self.obstacle_approach_factor = self.get_parameter('obstacle_approach_factor').value

        # Set up parameter callback
        self.add_on_set_parameters_callback(self.parameter_callback)

        # Publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)

        # Control timer
        self.control_timer = self.create_timer(0.1, self.control_loop)

        # Robot state
        self.scan_data = None

        self.get_logger().info('Adaptive Controller initialized')

    def parameter_callback(self, params):
        """Handle parameter updates"""
        result = SetParametersResult()
        result.successful = True

        for param in params:
            if param.name == 'linear_velocity':
                if param.type_ == Parameter.Type.PARAMETER_DOUBLE and param.value >= 0:
                    self.linear_velocity = param.value
                    self.get_logger().info(f'Updated linear_velocity to: {param.value}')
                else:
                    result.successful = False
                    result.reason = 'linear_velocity must be a non-negative double'
                    return result

            elif param.name == 'safety_distance':
                if param.type_ == Parameter.Type.PARAMETER_DOUBLE and param.value > 0:
                    self.safety_distance = param.value
                    self.get_logger().info(f'Updated safety_distance to: {param.value}')
                else:
                    result.successful = False
                    result.reason = 'safety_distance must be a positive double'
                    return result

        return result

    def scan_callback(self, msg):
        """Process laser scan data"""
        self.scan_data = msg

    def control_loop(self):
        """Main control loop"""
        if self.scan_data is None:
            return

        cmd = Twist()

        # Find minimum distance in front of robot (simplified forward sector)
        front_distances = self.scan_data.ranges[0:30] + self.scan_data.ranges[-30:]
        min_distance = min([d for d in front_distances if not math.isinf(d) and not math.isnan(d)], default=float('inf'))

        if min_distance < self.safety_distance:
            # Emergency stop
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        elif self.adaptive_enabled and min_distance < (self.safety_distance * self.obstacle_approach_factor):
            # Adaptive speed reduction based on obstacle proximity
            speed_factor = min_distance / (self.safety_distance * self.obstacle_approach_factor)
            cmd.linear.x = self.linear_velocity * speed_factor
            cmd.angular.z = 0.0  # Stop turning when approaching obstacles
        else:
            # Normal operation
            cmd.linear.x = self.linear_velocity
            cmd.angular.z = 0.0  # For this example, only move forward

        self.cmd_vel_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    controller = AdaptiveController()

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

## Parameter Management Strategies

### 1. Configuration Profiles

Create different parameter sets for different operating conditions:

```yaml
# config/indoor_config.yaml
/**:
  ros__parameters:
    max_velocity: 0.5
    safety_distance: 0.3
    localization_method: "amcl"

# config/outdoor_config.yaml
/**:
  ros__parameters:
    max_velocity: 1.5
    safety_distance: 1.0
    localization_method: "gps"
```

### 2. Parameter Presets

Allow switching between different parameter configurations:

```python
class PresetManager(Node):
    def __init__(self):
        super().__init__('preset_manager')

        # Predefined parameter sets
        self.presets = {
            'safe': {'max_velocity': 0.5, 'safety_distance': 0.5},
            'fast': {'max_velocity': 2.0, 'safety_distance': 1.0},
            'precise': {'max_velocity': 0.2, 'safety_distance': 0.3}
        }

    def load_preset(self, preset_name):
        """Load a predefined parameter set"""
        if preset_name in self.presets:
            preset = self.presets[preset_name]
            for param_name, param_value in preset.items():
                self.set_parameters([Parameter(param_name, value=param_value)])
            self.get_logger().info(f'Loaded preset: {preset_name}')
        else:
            self.get_logger().error(f'Unknown preset: {preset_name}')
```

## Key Takeaways

- Parameters provide dynamic configuration without code changes
- Use parameter callbacks for validation and response to changes
- Apply appropriate constraints and validation to parameters
- Document parameters with clear descriptions
- Use parameter files for configuration management
- Consider parameter synchronization across nodes
- Implement adaptive behavior based on parameter changes

## Next Steps

In the next chapter, we'll explore advanced communication patterns in ROS 2, including custom message types, complex data structures, and performance optimization techniques.