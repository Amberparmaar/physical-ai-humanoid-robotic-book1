---
sidebar_position: 2
title: Gazebo Fundamentals and World Creation
---

# Gazebo Fundamentals and World Creation

In this chapter, we'll explore the fundamentals of Gazebo, the leading robotics simulation environment. We'll cover world creation, physics simulation, and the essential concepts needed to build effective digital twins for robotic systems.

## Understanding Gazebo Architecture

Gazebo is built on a modular architecture that separates the simulation engine from the user interface and plugin system:

### Core Components

1. **Gazebo Server (gzserver)**: The headless simulation engine that handles physics, sensors, and plugin execution
2. **Gazebo Client (gzclient)**: The visualization interface that provides 3D rendering and user interaction
3. **Gazebo Transport**: A communication layer that enables message passing between components
4. **Plugin System**: Extensible architecture for custom sensors, controllers, and simulation features

### Physics Engine Integration

Gazebo supports multiple physics engines, with ODE (Open Dynamics Engine) being the most commonly used:

- **ODE**: Good balance of performance and accuracy for most robotics applications
- **Bullet**: More accurate contact simulation, good for complex interactions
- **Simbody**: High-fidelity simulation for complex articulated systems
- **DART**: Advanced collision detection and dynamics

## World Description Format (SDF)

Gazebo uses the Simulation Description Format (SDF) to define simulation worlds. SDF is an XML-based format that describes:

- World properties (gravity, magnetic field, etc.)
- Models and their properties
- Physics engine configuration
- Lighting and visual effects

### Basic World Structure

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="my_world">
    <!-- World properties -->
    <physics type="ode">
      <gravity>0 0 -9.8</gravity>
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
    </physics>

    <!-- Lighting -->
    <light name="sun" type="directional">
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
      </attenuation>
      <direction>-0.3 0.3 -0.9</direction>
    </light>

    <!-- Models -->
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <material>
            <ambient>0.7 0.7 0.7 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
            <specular>0.7 0.7 0.7 1</specular>
          </material>
        </visual>
      </link>
    </model>
  </world>
</sdf>
```

## Creating Custom Worlds

### Basic World with Obstacles

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="obstacle_course">
    <!-- Physics configuration -->
    <physics type="ode">
      <gravity>0 0 -9.8</gravity>
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>

    <!-- Sun light -->
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.3 0.3 -0.9</direction>
    </light>

    <!-- Ground plane -->
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <material>
            <ambient>0.7 0.7 0.7 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
            <specular>0.7 0.7 0.7 1</specular>
          </material>
        </visual>
      </link>
    </model>

    <!-- Obstacles -->
    <model name="wall_1">
      <pose>0 5 1 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>10 0.2 2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>10 0.2 2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.1 0.1 1</ambient>
            <diffuse>0.8 0.1 0.1 1</diffuse>
            <specular>0.8 0.1 0.1 1</specular>
          </material>
        </visual>
      </link>
    </model>

    <model name="box_obstacle">
      <pose>3 0 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.1 0.1 0.8 1</ambient>
            <diffuse>0.1 0.1 0.8 1</diffuse>
            <specular>0.1 0.1 0.8 1</specular>
          </material>
        </visual>
      </link>
    </model>
  </world>
</sdf>
```

## Model Description Format

Models in Gazebo are also defined using SDF and include:

### Link Elements
- **Collision**: Defines physical properties for collision detection
- **Visual**: Defines how the model appears in the simulation
- **Inertial**: Defines mass, center of mass, and moment of inertia

### Joint Elements
- **Revolute**: Rotational joint with one degree of freedom
- **Prismatic**: Linear joint with one degree of freedom
- **Fixed**: Rigid connection between links
- **Continuous**: Rotational joint without limits
- **Prismatic**: Linear joint with limits

### Example Model: Simple Robot

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <model name="simple_robot">
    <link name="chassis">
      <pose>0 0 0.1 0 0 0</pose>
      <inertial>
        <mass>1.0</mass>
        <inertia>
          <ixx>0.01</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.01</iyy>
          <iyz>0</iyz>
          <izz>0.01</izz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry>
          <box>
            <size>0.5 0.3 0.1</size>
          </box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box>
            <size>0.5 0.3 0.1</size>
          </box>
        </geometry>
        <material>
          <ambient>0.8 0.8 0.2 1</ambient>
          <diffuse>0.8 0.8 0.2 1</diffuse>
          <specular>0.8 0.8 0.2 1</specular>
        </material>
      </visual>
    </link>

    <link name="wheel_left">
      <pose>-0.15 0.2 0 0 0 0</pose>
      <inertial>
        <mass>0.2</mass>
        <inertia>
          <ixx>0.001</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.001</iyy>
          <iyz>0</iyz>
          <izz>0.001</izz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry>
          <cylinder>
            <radius>0.1</radius>
            <length>0.05</length>
          </cylinder>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>0.1</radius>
            <length>0.05</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0.2 0.2 0.2 1</ambient>
          <diffuse>0.2 0.2 0.2 1</diffuse>
          <specular>0.2 0.2 0.2 1</specular>
        </material>
      </visual>
    </link>

    <joint name="left_wheel_joint" type="revolute">
      <parent>chassis</parent>
      <child>wheel_left</child>
      <axis>
        <xyz>0 1 0</xyz>
      </axis>
    </joint>
  </model>
</sdf>
```

## Physics Configuration

### Time Step and Real-time Factors

```xml
<physics type="ode">
  <max_step_size>0.001</max_step_size>  <!-- Simulation time step (seconds) -->
  <real_time_factor>1</real_time_factor>  <!-- Speed relative to real time -->
  <real_time_update_rate>1000</real_time_update_rate>  <!-- Hz -->
</physics>
```

### Contact and Friction Parameters

```xml
<physics type="ode">
  <ode>
    <solver>
      <type>quick</type>
      <iters>10</iters>
      <sor>1.3</sor>
    </solver>
    <constraints>
      <cfm>0.0</cfm>
      <erp>0.2</erp>
      <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
</physics>
```

## Working with Gazebo Programmatically

### Launching Gazebo with Custom Worlds

```bash
# Launch Gazebo with a custom world
gzserver /path/to/my_world.world

# Launch with GUI
gzserver /path/to/my_world.world &
gzclient
```

### Using Gazebo with ROS 2

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import subprocess
import time

class GazeboManager(Node):
    def __init__(self):
        super().__init__('gazebo_manager')

        # Launch Gazebo programmatically
        self.launch_gazebo()

        # Publisher for simulation commands
        self.sim_cmd_pub = self.create_publisher(String, 'sim_commands', 10)

        self.get_logger().info('Gazebo Manager initialized')

    def launch_gazebo(self):
        """Launch Gazebo with a custom world"""
        world_path = "/path/to/my_custom_world.sdf"

        # Launch gazebo server
        self.gazebo_process = subprocess.Popen([
            'gzserver',
            world_path,
            '--verbose'  # For debugging
        ])

        # Wait for Gazebo to start
        time.sleep(3)

        self.get_logger().info('Gazebo server launched')

    def shutdown_gazebo(self):
        """Properly shutdown Gazebo"""
        if hasattr(self, 'gazebo_process'):
            self.gazebo_process.terminate()
            self.gazebo_process.wait()
```

## Advanced World Features

### Terrain Generation

```xml
<model name="uneven_terrain">
  <static>true</static>
  <link name="link">
    <collision name="collision">
      <geometry>
        <heightmap>
          <uri>model://my_terrain/heightmap.png</uri>
          <size>100 100 10</size>
        </heightmap>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <heightmap>
          <uri>model://my_terrain/heightmap.png</uri>
          <size>100 100 10</size>
        </heightmap>
      </geometry>
      <material>
        <script>
          <uri>file://media/materials/scripts/gazebo.material</uri>
          <name>Gazebo/Dirt</name>
        </script>
      </material>
    </visual>
  </link>
</model>
```

### Particle Effects and Atmospherics

```xml
<world name="outdoor_world">
  <!-- Fog effects -->
  <scene>
    <fog type="linear">
      <color>0.8 0.8 0.8</color>
      <density>0.01</density>
      <range>1 100</range>
    </fog>
  </scene>

  <!-- Wind effects -->
  <wind>
    <linear_velocity>0.5 0 0</linear_velocity>
  </wind>
</world>
```

## Performance Optimization

### Level of Detail (LOD)

For complex simulations, consider using simplified models at greater distances:

```xml
<visual name="detailed_visual">
  <geometry>
    <mesh>
      <uri>model://robot/meshes/detailed.dae</uri>
    </mesh>
  </geometry>
  <!-- Use simpler collision geometry -->
  <collision name="simple_collision">
    <geometry>
      <box>
        <size>0.5 0.3 0.2</size>
      </box>
    </geometry>
  </collision>
</visual>
```

### Simulation Parameters for Performance

```xml
<physics type="ode">
  <max_step_size>0.01</max_step_size>  <!-- Larger steps for performance -->
  <real_time_factor>1.0</real_time_factor>  <!-- Match real-time for stability -->
  <real_time_update_rate>100</real_time_update_rate>  <!-- Lower rate for performance -->
  <ode>
    <solver>
      <iters>20</iters>  <!-- Balance between accuracy and performance -->
    </solver>
  </ode>
</physics>
```

## Best Practices for World Creation

### 1. Start Simple
Begin with basic geometric shapes and gradually add complexity:

```xml
<!-- Start with a simple box world -->
<model name="simple_obstacle">
  <link name="link">
    <collision name="collision">
      <geometry>
        <box>
          <size>1 1 1</size>
        </box>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <box>
          <size>1 1 1</size>
        </box>
      </geometry>
    </visual>
  </link>
</model>
```

### 2. Use Proper Units
Always use consistent units (typically meters for length, kilograms for mass):

```xml
<inertial>
  <mass>1.5</mass>  <!-- 1.5 kg -->
  <pose>0 0 0.05 0 0 0</pose>  <!-- 5 cm height -->
</inertial>
```

### 3. Validate Physics Properties
Ensure mass, inertia, and friction values are realistic:

```xml
<collision name="wheel_collision">
  <surface>
    <friction>
      <ode>
        <mu>1.0</mu>  <!-- Coefficient of friction -->
        <mu2>1.0</mu2>
      </ode>
    </friction>
  </surface>
</collision>
```

## Practical Example: Warehouse Simulation

Let's create a complete warehouse simulation world:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="warehouse">
    <!-- Physics -->
    <physics type="ode">
      <gravity>0 0 -9.8</gravity>
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>

    <!-- Lighting -->
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.3 0.3 -0.9</direction>
    </light>

    <!-- Artificial lights -->
    <light name="light_1" type="point">
      <pose>0 0 5 0 0 0</pose>
      <diffuse>0.9 0.9 0.9 1</diffuse>
      <specular>0.1 0.1 0.1 1</specular>
      <attenuation>
        <range>10</range>
      </attenuation>
    </light>

    <!-- Ground -->
    <model name="floor">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>20 20 0.1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>20 20 0.1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.7 0.7 0.7 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Warehouse racks -->
    <model name="rack_1">
      <pose>5 0 1 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>2 0.5 2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>2 0.5 2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.5 0.5 0.5 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Charging station -->
    <model name="charging_station">
      <pose>-5 -5 0.2 0 0 0</pose>
      <link name="base">
        <collision name="collision">
          <geometry>
            <box>
              <size>1 1 0.4</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1 1 0.4</size>
            </box>
          </geometry>
          <material>
            <ambient>0.1 0.8 0.1 1</ambient>
            <diffuse>0.1 0.8 0.1 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
  </world>
</sdf>
```

## Key Takeaways

- Gazebo uses SDF (Simulation Description Format) for world and model descriptions
- Physics configuration affects simulation accuracy and performance
- Proper mass, inertia, and friction values are crucial for realistic simulation
- Start with simple worlds and gradually add complexity
- Use appropriate time steps and real-time factors for your application
- Consider performance implications when creating complex worlds

## Next Steps

In the next chapter, we'll explore robot modeling and URDF integration, learning how to create detailed robot models that can be used in Gazebo simulations.