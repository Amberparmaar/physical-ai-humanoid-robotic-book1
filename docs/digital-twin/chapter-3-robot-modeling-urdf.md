---
sidebar_position: 3
title: Robot Modeling and URDF Integration
---

# Robot Modeling and URDF Integration

In this chapter, we'll explore Unified Robot Description Format (URDF), the standard for representing robot models in ROS. We'll learn how to create detailed robot models that can be used in Gazebo simulations, connecting the digital twin to realistic physical representations.

## Understanding URDF

URDF (Unified Robot Description Format) is an XML-based format that describes robot models in ROS. It defines:

- **Kinematic structure**: Joint connections and degrees of freedom
- **Geometric properties**: Shape, size, and visual appearance
- **Physical properties**: Mass, inertia, and friction
- **Sensors and actuators**: Mounting points and specifications

### Basic URDF Structure

```xml
<?xml version="1.0"?>
<robot name="my_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Base link - the root of the kinematic tree -->
  <link name="base_link">
    <inertial>
      <mass value="1.0" />
      <origin xyz="0 0 0" rpy="0 0 0" />
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01" />
    </inertial>

    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <box size="0.5 0.3 0.1" />
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1" />
      </material>
    </visual>

    <collision>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <box size="0.5 0.3 0.1" />
      </geometry>
    </collision>
  </link>

  <!-- Additional links connected via joints -->
  <link name="wheel_left">
    <inertial>
      <mass value="0.2" />
      <origin xyz="0 0 0" rpy="0 0 0" />
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001" />
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="1.5708 0 0" />
      <geometry>
        <cylinder radius="0.1" length="0.05" />
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="1.5708 0 0" />
      <geometry>
        <cylinder radius="0.1" length="0.05" />
      </geometry>
    </collision>
  </link>

  <!-- Joint connecting links -->
  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link" />
    <child link="wheel_left" />
    <origin xyz="-0.15 0.2 0" rpy="0 0 0" />
    <axis xyz="0 1 0" />
  </joint>
</robot>
```

## URDF Elements in Detail

### Link Elements

Each link represents a rigid body in the robot:

```xml
<link name="link_name">
  <!-- Physical properties -->
  <inertial>
    <mass value="1.0" />
    <origin xyz="0 0 0" rpy="0 0 0" />
    <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01" />
  </inertial>

  <!-- Visual representation -->
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0" />
    <geometry>
      <!-- Can be box, cylinder, sphere, or mesh -->
      <box size="0.5 0.3 0.1" />
    </geometry>
    <material name="red">
      <color rgba="1 0 0 1" />
    </material>
  </visual>

  <!-- Collision geometry -->
  <collision>
    <origin xyz="0 0 0" rpy="0 0 0" />
    <geometry>
      <box size="0.5 0.3 0.1" />
    </geometry>
  </collision>
</link>
```

### Joint Elements

Joints define the connection between links:

```xml
<joint name="joint_name" type="joint_type">
  <parent link="parent_link_name" />
  <child link="child_link_name" />
  <origin xyz="0 0 0" rpy="0 0 0" />
  <axis xyz="1 0 0" />

  <!-- Joint limits (for revolute and prismatic joints) -->
  <limit lower="-1.57" upper="1.57" effort="10" velocity="1" />

  <!-- Joint dynamics -->
  <dynamics damping="0.1" friction="0.0" />
</joint>
```

## Joint Types

### 1. Fixed Joint
Rigid connection with no degrees of freedom:

```xml
<joint name="fixed_joint" type="fixed">
  <parent link="base_link" />
  <child link="sensor_mount" />
  <origin xyz="0.1 0 0.1" rpy="0 0 0" />
</joint>
```

### 2. Revolute Joint
Single rotational degree of freedom with limits:

```xml
<joint name="revolute_joint" type="revolute">
  <parent link="upper_arm" />
  <child link="forearm" />
  <origin xyz="0 0 0.3" rpy="0 0 0" />
  <axis xyz="0 0 1" />
  <limit lower="-1.57" upper="1.57" effort="100" velocity="1.0" />
  <dynamics damping="0.5" friction="0.1" />
</joint>
```

### 3. Continuous Joint
Continuous rotation (like a wheel):

```xml
<joint name="continuous_joint" type="continuous">
  <parent link="base_link" />
  <child link="wheel" />
  <origin xyz="0 0.2 0" rpy="0 0 0" />
  <axis xyz="0 1 0" />
  <dynamics damping="0.1" friction="0.05" />
</joint>
```

### 4. Prismatic Joint
Linear motion with limits:

```xml
<joint name="prismatic_joint" type="prismatic">
  <parent link="base" />
  <child link="slider" />
  <origin xyz="0 0 0.1" rpy="0 0 0" />
  <axis xyz="0 0 1" />
  <limit lower="0" upper="0.5" effort="50" velocity="0.5" />
</joint>
```

## Geometry Types

### Primitive Shapes

```xml
<!-- Box -->
<geometry>
  <box size="0.5 0.3 0.1" />
</geometry>

<!-- Cylinder -->
<geometry>
  <cylinder radius="0.1" length="0.2" />
</geometry>

<!-- Sphere -->
<geometry>
  <sphere radius="0.05" />
</geometry>
```

### Mesh Geometry

```xml
<geometry>
  <mesh filename="package://my_robot_description/meshes/link.dae" scale="1 1 1" />
</geometry>
```

## Materials and Colors

```xml
<!-- Define materials -->
<material name="red">
  <color rgba="1 0 0 1" />
</material>

<material name="blue">
  <color rgba="0 0 1 1" />
</material>

<!-- Or use textures -->
<material name="texture_material">
  <texture filename="package://my_robot_description/textures/texture.png" />
</material>
```

## Complete Robot Example: Differential Drive Robot

```xml
<?xml version="1.0"?>
<robot name="diff_drive_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Base footprint (for navigation) -->
  <link name="base_footprint">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <box size="0.001 0.001 0.001" />
      </geometry>
    </visual>
  </link>

  <!-- Base link -->
  <joint name="base_joint" type="fixed">
    <parent link="base_footprint" />
    <child link="base_link" />
    <origin xyz="0 0 0.1" rpy="0 0 0" />
  </joint>

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
      <material name="light_grey">
        <color rgba="0.7 0.7 0.7 1" />
      </material>
    </visual>

    <collision>
      <origin xyz="0 0 0.1" rpy="0 0 0" />
      <geometry>
        <box size="0.5 0.3 0.2" />
      </geometry>
    </collision>
  </link>

  <!-- Left wheel -->
  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link" />
    <child link="left_wheel_link" />
    <origin xyz="0 0.2 0" rpy="-1.5708 0 0" />
    <axis xyz="0 0 1" />
    <limit effort="100" velocity="100" />
    <dynamics damping="1.0" friction="1.0" />
  </joint>

  <link name="left_wheel_link">
    <inertial>
      <mass value="2.0" />
      <origin xyz="0 0 0" rpy="0 0 0" />
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.02" />
    </inertial>

    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <cylinder radius="0.1" length="0.05" />
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1" />
      </material>
    </visual>

    <collision>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <cylinder radius="0.1" length="0.05" />
      </geometry>
    </collision>
  </link>

  <!-- Right wheel -->
  <joint name="right_wheel_joint" type="continuous">
    <parent link="base_link" />
    <child link="right_wheel_link" />
    <origin xyz="0 -0.2 0" rpy="1.5708 0 0" />
    <axis xyz="0 0 -1" />
    <limit effort="100" velocity="100" />
    <dynamics damping="1.0" friction="1.0" />
  </joint>

  <link name="right_wheel_link">
    <inertial>
      <mass value="2.0" />
      <origin xyz="0 0 0" rpy="0 0 0" />
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.02" />
    </inertial>

    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <cylinder radius="0.1" length="0.05" />
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1" />
      </material>
    </visual>

    <collision>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <cylinder radius="0.1" length="0.05" />
      </geometry>
    </collision>
  </link>

  <!-- Camera -->
  <joint name="camera_joint" type="fixed">
    <parent link="base_link" />
    <child link="camera_link" />
    <origin xyz="0.2 0 0.1" rpy="0 0 0" />
  </joint>

  <link name="camera_link">
    <inertial>
      <mass value="0.1" />
      <origin xyz="0 0 0" rpy="0 0 0" />
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001" />
    </inertial>

    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <box size="0.05 0.05 0.05" />
      </geometry>
      <material name="red">
        <color rgba="1 0 0 1" />
      </material>
    </visual>

    <collision>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <box size="0.05 0.05 0.05" />
      </geometry>
    </collision>
  </link>
</robot>
```

## URDF with Gazebo Integration

To make URDF models work properly in Gazebo, add Gazebo-specific tags:

```xml
<?xml version="1.0"?>
<robot name="gazebo_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Include Gazebo materials -->
  <gazebo reference="base_link">
    <material>Gazebo/Blue</material>
    <mu1>0.2</mu1>
    <mu2>0.2</mu2>
    <self_collide>false</self_collide>
  </gazebo>

  <!-- Sensor definition -->
  <gazebo reference="camera_link">
    <sensor type="camera" name="camera1">
      <update_rate>30.0</update_rate>
      <camera name="head">
        <horizontal_fov>1.3962634</horizontal_fov>
        <image>
          <width>800</width>
          <height>600</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.02</near>
          <far>300</far>
        </clip>
      </camera>
      <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
        <frame_name>camera_optical_frame</frame_name>
      </plugin>
    </sensor>
  </gazebo>

  <!-- Wheel transmission for differential drive -->
  <transmission name="left_wheel_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="left_wheel_joint">
      <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
    </joint>
    <actuator name="left_wheel_motor">
      <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="right_wheel_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="right_wheel_joint">
      <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
    </joint>
    <actuator name="right_wheel_motor">
      <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>
</robot>
```

## Xacro: XML Macros for URDF

Xacro allows creating reusable URDF components:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="xacro_robot">
  <!-- Properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="wheel_radius" value="0.1" />
  <xacro:property name="wheel_width" value="0.05" />
  <xacro:property name="base_mass" value="10.0" />

  <!-- Macro for creating wheels -->
  <xacro:macro name="wheel" params="prefix parent xyz rpy">
    <joint name="${prefix}_wheel_joint" type="continuous">
      <parent link="${parent}" />
      <child link="${prefix}_wheel_link" />
      <origin xyz="${xyz}" rpy="${rpy}" />
      <axis xyz="0 1 0" />
      <limit effort="100" velocity="100" />
      <dynamics damping="1.0" friction="1.0" />
    </joint>

    <link name="${prefix}_wheel_link">
      <inertial>
        <mass value="2.0" />
        <origin xyz="0 0 0" rpy="0 0 0" />
        <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.02" />
      </inertial>

      <visual>
        <origin xyz="0 0 0" rpy="0 0 0" />
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}" />
        </geometry>
        <material name="black">
          <color rgba="0 0 0 1" />
        </material>
      </visual>

      <collision>
        <origin xyz="0 0 0" rpy="0 0 0" />
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}" />
        </geometry>
      </collision>
    </link>
  </xacro:macro>

  <!-- Use the macro -->
  <link name="base_link">
    <inertial>
      <mass value="${base_mass}" />
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

  <!-- Create wheels using the macro -->
  <xacro:wheel prefix="left" parent="base_link" xyz="0 0.2 0" rpy="0 0 0" />
  <xacro:wheel prefix="right" parent="base_link" xyz="0 -0.2 0" rpy="0 0 0" />
</robot>
```

## Advanced URDF Features

### Gazebo Plugins

```xml
<!-- Differential drive plugin -->
<gazebo>
  <plugin name="differential_drive_controller" filename="libgazebo_ros_diff_drive.so">
    <leftJoint>left_wheel_joint</leftJoint>
    <rightJoint>right_wheel_joint</rightJoint>
    <wheelSeparation>0.4</wheelSeparation>
    <wheelDiameter>0.2</wheelDiameter>
    <commandTopic>cmd_vel</commandTopic>
    <odometryTopic>odom</odometryTopic>
    <odometryFrame>odom</odometryFrame>
    <robotBaseFrame>base_link</robotBaseFrame>
    <publishWheelTF>true</publishWheelTF>
    <publishOdomTF>true</publishOdomTF>
    <odometrySource>world</odometrySource>
  </plugin>
</gazebo>
```

### Sensor Integration

```xml
<!-- IMU sensor -->
<gazebo reference="imu_link">
  <sensor type="imu" name="imu_sensor">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <visualize>true</visualize>
    <imu>
      <noise>
        <type>gaussian</type>
        <rate>
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
          <bias_mean>0.0000075</bias_mean>
          <bias_stddev>0.0000008</bias_stddev>
        </rate>
        <accel>
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
          <bias_mean>0.1</bias_mean>
          <bias_stddev>0.001</bias_stddev>
        </accel>
      </noise>
    </imu>
  </sensor>
</gazebo>
```

## Working with URDF in ROS 2

### Launching Robot in Gazebo

```python
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Get URDF file path
    pkg_path = os.path.join(get_package_share_directory('my_robot_description'))
    urdf_file = os.path.join(pkg_path, 'urdf', 'my_robot.urdf')

    # Launch Gazebo with the robot
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('gazebo_ros'),
            '/launch',
            '/gazebo.launch.py'
        ])
    )

    # Spawn robot in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-file', urdf_file,
            '-entity', 'my_robot',
            '-x', '0', '-y', '0', '-z', '0.1'
        ],
        output='screen'
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'robot_description': open(urdf_file).read()
        }]
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation (Gazebo) clock if true'
        ),
        gazebo,
        robot_state_publisher,
        spawn_entity
    ])
```

## Validation and Testing

### URDF Validation

```bash
# Check URDF syntax
check_urdf /path/to/robot.urdf

# Get URDF info
urdf_to_graphiz /path/to/robot.urdf
```

### Visualization

```bash
# Visualize robot in RViz
ros2 run rviz2 rviz2

# Or use robot state publisher with joint state publisher GUI
ros2 run robot_state_publisher robot_state_publisher --ros-args -p robot_description:='$(find my_robot_description)/urdf/my_robot.urdf'
ros2 run joint_state_publisher_gui joint_state_publisher_gui
```

## Best Practices for Robot Modeling

### 1. Proper Inertial Properties
Calculate realistic mass and inertia values:

```xml
<!-- Example for a cylindrical object -->
<inertial>
  <mass value="0.5" />
  <origin xyz="0 0 0" rpy="0 0 0" />
  <!-- For cylinder: Ixx = Iyy = m*(3*r² + h²)/12, Izz = m*r²/2 -->
  <inertia ixx="0.0013" ixy="0" ixz="0" iyy="0.0013" iyz="0" izz="0.00125" />
</inertial>
```

### 2. Appropriate Collision Geometry
Use simpler geometry for collision than visual:

```xml
<!-- Visual: detailed mesh -->
<visual>
  <geometry>
    <mesh filename="complex_shape.dae" />
  </geometry>
</visual>

<!-- Collision: simplified box -->
<collision>
  <geometry>
    <box size="0.1 0.1 0.1" />
  </geometry>
</collision>
```

### 3. Consistent Units
Always use meters for length, kilograms for mass, and radians for angles.

### 4. Proper Joint Limits
Set realistic limits based on physical constraints:

```xml
<joint name="arm_joint" type="revolute">
  <limit lower="-1.57" upper="1.57" effort="100" velocity="1.0" />
</joint>
```

## Key Takeaways

- URDF defines robot structure, geometry, and physics properties
- Proper inertial properties are crucial for realistic simulation
- Gazebo-specific tags enable advanced simulation features
- Xacro macros improve reusability and maintainability
- Joint types determine robot mobility and functionality
- Validation tools help ensure correct URDF structure

## Next Steps

In the next chapter, we'll explore sensor simulation and perception in Gazebo, learning how to create realistic sensor models that accurately represent physical sensors in the digital twin.