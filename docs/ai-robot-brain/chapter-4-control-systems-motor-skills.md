---
sidebar_position: 4
title: Control Systems and Motor Skills
---

# Control Systems and Motor Skills

In this chapter, we'll explore how AI robot brains translate high-level plans and decisions into precise physical actions through sophisticated control systems. We'll cover the fundamental control algorithms, motor skill learning, and the integration of perception and action in real-time control systems.

## Understanding Robot Control Systems

Robot control systems bridge the gap between abstract planning and physical execution. They can be categorized into:

### 1. Open-Loop Control
- Commands sent without feedback
- Suitable for predictable, repeatable tasks
- Simple but lacks adaptability

### 2. Closed-Loop Control (Feedback Control)
- Uses sensor feedback to adjust commands
- Adapts to disturbances and uncertainties
- More robust and accurate

### 3. Feedforward Control
- Anticipates required actions based on planned trajectories
- Works in conjunction with feedback control
- Improves response time and accuracy

## Classical Control Algorithms

### PID Controller Implementation

```python
import numpy as np
from typing import Tuple

class PIDController:
    def __init__(self, kp: float, ki: float, kd: float, dt: float = 0.01):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.dt = dt  # Time step

        self.error_sum = 0.0
        self.prev_error = 0.0
        self.prev_output = 0.0

    def update(self, setpoint: float, measured_value: float) -> float:
        """Update PID controller and return control output"""
        # Calculate error
        error = setpoint - measured_value

        # Proportional term
        p_term = self.kp * error

        # Integral term
        self.error_sum += error * self.dt
        i_term = self.ki * self.error_sum

        # Derivative term
        derivative = (error - self.prev_error) / self.dt if self.dt > 0 else 0.0
        d_term = self.kd * derivative

        # Calculate output
        output = p_term + i_term + d_term

        # Apply output limits to prevent windup
        output = np.clip(output, -100.0, 100.0)

        # Store values for next iteration
        self.prev_error = error

        return output

    def reset(self):
        """Reset the controller"""
        self.error_sum = 0.0
        self.prev_error = 0.0
        self.prev_output = 0.0

class JointController:
    def __init__(self, joint_name: str, kp: float, ki: float, kd: float):
        self.joint_name = joint_name
        self.position_controller = PIDController(kp, ki, kd)
        self.velocity_controller = PIDController(kp/10, ki/10, kd/10)  # Lower gains for velocity

        self.current_position = 0.0
        self.current_velocity = 0.0
        self.desired_position = 0.0
        self.desired_velocity = 0.0

    def update_position_control(self, dt: float) -> float:
        """Update position control loop"""
        position_error = self.desired_position - self.current_position
        velocity_command = self.position_controller.update(0.0, position_error)

        # Use velocity command as feedforward term
        self.desired_velocity = velocity_command

        return self.update_velocity_control(dt)

    def update_velocity_control(self, dt: float) -> float:
        """Update velocity control loop"""
        velocity_error = self.desired_velocity - self.current_velocity
        torque_command = self.velocity_controller.update(0.0, velocity_error)

        return torque_command

    def set_desired_position(self, position: float):
        """Set desired joint position"""
        self.desired_position = position

    def set_desired_velocity(self, velocity: float):
        """Set desired joint velocity"""
        self.desired_velocity = velocity

    def get_state(self) -> Tuple[float, float]:
        """Get current position and velocity"""
        return self.current_position, self.current_velocity

    def set_state(self, position: float, velocity: float):
        """Set current position and velocity"""
        self.current_position = position
        self.current_velocity = velocity
```

### Advanced PID Techniques

```python
class AdvancedPIDController:
    def __init__(self, kp: float, ki: float, kd: float, dt: float = 0.01):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt

        # Anti-windup parameters
        self.output_limits = (-100.0, 100.0)
        self.integral_limits = (-50.0, 50.0)

        # Derivative filtering
        self.derivative_filter = 0.1  # Low-pass filter coefficient

        self.error_sum = 0.0
        self.prev_error = 0.0
        self.prev_derivative = 0.0
        self.prev_output = 0.0

    def update(self, setpoint: float, measured_value: float) -> float:
        """Update PID controller with advanced features"""
        error = setpoint - measured_value

        # Proportional term
        p_term = self.kp * error

        # Integral term with anti-windup
        self.error_sum += error * self.dt
        # Clamp integral term to prevent windup
        self.error_sum = np.clip(self.error_sum, self.integral_limits[0], self.integral_limits[1])
        i_term = self.ki * self.error_sum

        # Derivative term with filtering
        raw_derivative = (error - self.prev_error) / self.dt if self.dt > 0 else 0.0
        # Low-pass filter the derivative to reduce noise
        filtered_derivative = (self.derivative_filter * raw_derivative +
                              (1 - self.derivative_filter) * self.prev_derivative)
        d_term = self.kd * filtered_derivative

        # Store for next iteration
        self.prev_error = error
        self.prev_derivative = filtered_derivative

        # Calculate output
        output = p_term + i_term + d_term

        # Apply output limits
        clamped_output = np.clip(output, self.output_limits[0], self.output_limits[1])

        # Anti-windup: reduce integral term if output is saturated
        if clamped_output != output:
            # Output is saturated, reduce integral term
            self.error_sum -= (output - clamped_output) / self.ki

        self.prev_output = clamped_output

        return clamped_output

    def tune_gains(self, method: str = 'ziegler-nichols'):
        """Auto-tune PID gains using various methods"""
        if method == 'ziegler-nichols':
            # Ziegler-Nichols tuning rules (simplified)
            # In practice, you'd run an actual tuning procedure
            ultimate_gain = 10.0  # This would be determined experimentally
            ultimate_period = 2.0  # This would be determined experimentally

            self.kp = 0.6 * ultimate_gain
            self.ki = 2 * self.kp / ultimate_period
            self.kd = self.kp * ultimate_period / 8
```

## Trajectory Generation and Following

### Minimum Jerk Trajectory

```python
import numpy as np
from typing import List, Tuple

class MinimumJerkTrajectory:
    def __init__(self, start_pos: float, end_pos: float, duration: float):
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.duration = duration
        self.coeffs = self._calculate_coefficients()

    def _calculate_coefficients(self) -> np.ndarray:
        """Calculate coefficients for minimum jerk trajectory"""
        # Minimum jerk trajectory: x(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
        # Subject to: x(0) = start, x(T) = end, x'(0) = 0, x'(T) = 0, x''(0) = 0, x''(T) = 0

        T = self.duration
        x0 = self.start_pos
        xT = self.end_pos

        # Solve for coefficients
        a0 = x0
        a1 = 0  # zero initial velocity
        a2 = 0  # zero initial acceleration
        a3 = (20 * (xT - x0)) / (2 * T**3)
        a4 = (-30 * (xT - x0)) / (2 * T**4)
        a5 = (12 * (xT - x0)) / (2 * T**5)

        return np.array([a0, a1, a2, a3, a4, a5])

    def get_position(self, t: float) -> float:
        """Get position at time t"""
        if t <= 0:
            return self.start_pos
        elif t >= self.duration:
            return self.end_pos

        t_power = np.array([1, t, t**2, t**3, t**4, t**5])
        return np.dot(self.coeffs, t_power)

    def get_velocity(self, t: float) -> float:
        """Get velocity at time t"""
        if t <= 0 or t >= self.duration:
            return 0.0

        t_power = np.array([0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4])
        return np.dot(self.coeffs, t_power)

    def get_acceleration(self, t: float) -> float:
        """Get acceleration at time t"""
        if t <= 0 or t >= self.duration:
            return 0.0

        t_power = np.array([0, 0, 2, 6*t, 12*t**2, 20*t**3])
        return np.dot(self.coeffs, t_power)

class TrajectoryFollower:
    def __init__(self, dt: float = 0.01):
        self.dt = dt
        self.current_time = 0.0
        self.trajectory = None

    def set_trajectory(self, trajectory: MinimumJerkTrajectory):
        """Set the trajectory to follow"""
        self.trajectory = trajectory
        self.current_time = 0.0

    def update(self) -> Tuple[float, float, float]:
        """Update and get desired position, velocity, acceleration"""
        if self.trajectory is None:
            return 0.0, 0.0, 0.0

        pos = self.trajectory.get_position(self.current_time)
        vel = self.trajectory.get_velocity(self.current_time)
        acc = self.trajectory.get_acceleration(self.current_time)

        self.current_time += self.dt

        return pos, vel, acc

    def is_complete(self) -> bool:
        """Check if trajectory following is complete"""
        if self.trajectory is None:
            return True
        return self.current_time >= self.trajectory.duration
```

### Multi-Joint Trajectory Generation

```python
class MultiJointTrajectory:
    def __init__(self, joint_names: List[str], start_positions: List[float],
                 end_positions: List[float], duration: float):
        self.joint_names = joint_names
        self.duration = duration

        # Create trajectory for each joint
        self.trajectories = {}
        for i, joint_name in enumerate(joint_names):
            trajectory = MinimumJerkTrajectory(
                start_positions[i], end_positions[i], duration
            )
            self.trajectories[joint_name] = trajectory

    def get_joint_position(self, joint_name: str, t: float) -> float:
        """Get position for specific joint at time t"""
        return self.trajectories[joint_name].get_position(t)

    def get_joint_velocity(self, joint_name: str, t: float) -> float:
        """Get velocity for specific joint at time t"""
        return self.trajectories[joint_name].get_velocity(t)

    def get_joint_acceleration(self, joint_name: str, t: float) -> float:
        """Get acceleration for specific joint at time t"""
        return self.trajectories[joint_name].get_acceleration(t)

    def get_all_positions(self, t: float) -> List[float]:
        """Get positions for all joints at time t"""
        return [self.get_joint_position(name, t) for name in self.joint_names]

    def get_all_velocities(self, t: float) -> List[float]:
        """Get velocities for all joints at time t"""
        return [self.get_joint_velocity(name, t) for name in self.joint_names]

    def get_all_accelerations(self, t: float) -> List[float]:
        """Get accelerations for all joints at time t"""
        return [self.get_joint_acceleration(name, t) for name in self.joint_names]
```

## Adaptive Control Systems

### Model Reference Adaptive Control (MRAC)

```python
class MRACController:
    def __init__(self, reference_model_params: np.ndarray, dt: float = 0.01):
        self.dt = dt
        self.reference_model_params = reference_model_params
        self.theta = np.zeros(len(reference_model_params))  # Adaptive parameters
        self.gamma = 0.1  # Adaptation gain
        self.sigma = 0.01  # Sigma modification for stability

        self.prev_error = 0.0
        self.prev_reference = 0.0

    def update(self, reference_input: float, plant_output: float) -> float:
        """Update MRAC controller"""
        # Reference model output
        reference_output = self._reference_model(reference_input)

        # Tracking error
        error = reference_output - plant_output

        # Adaptive law with sigma modification
        phi = self._regression_vector(reference_input, plant_output)
        self.theta += self.dt * self.gamma * (error + self.sigma * self.theta)

        # Control law
        control_signal = np.dot(phi, self.theta)

        self.prev_error = error
        self.prev_reference = reference_output

        return control_signal

    def _reference_model(self, input_signal: float) -> float:
        """Reference model (first-order system)"""
        # y_dot = -a*y + b*u
        a, b = self.reference_model_params
        output = -a * self.prev_reference + b * input_signal
        return output * self.dt + self.prev_reference

    def _regression_vector(self, input_signal: float, output_signal: float) -> np.ndarray:
        """Regression vector for parameter adaptation"""
        # For a simple adaptive controller
        return np.array([input_signal, output_signal, 1.0])  # [u, y, 1]
```

### Self-Organizing Maps for Control

```python
class SOMController:
    def __init__(self, input_dim: int, output_dim: int, grid_size: int = 10):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.grid_size = grid_size

        # Initialize SOM weights randomly
        self.weights = np.random.random((grid_size, grid_size, input_dim))
        self.output_map = np.random.random((grid_size, grid_size, output_dim))

        self.learning_rate = 0.1
        self.neighborhood_radius = grid_size / 2

    def find_bmu(self, input_vector: np.ndarray) -> Tuple[int, int]:
        """Find Best Matching Unit (BMU)"""
        distances = np.linalg.norm(
            self.weights - input_vector.reshape(1, 1, -1), axis=2
        )
        bmu_idx = np.unravel_index(np.argmin(distances), distances.shape)
        return bmu_idx

    def update_weights(self, input_vector: np.ndarray, bmu: Tuple[int, int]):
        """Update weights based on input and BMU"""
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                distance = np.sqrt((i - bmu[0])**2 + (j - bmu[1])**2)
                if distance <= self.neighborhood_radius:
                    influence = np.exp(-distance**2 / (2 * self.neighborhood_radius**2))
                    self.weights[i, j] += (self.learning_rate * influence *
                                         (input_vector - self.weights[i, j]))

    def get_output(self, input_vector: np.ndarray) -> np.ndarray:
        """Get control output for given input"""
        bmu = self.find_bmu(input_vector)
        return self.output_map[bmu[0], bmu[1]]

    def train(self, input_data: List[np.ndarray], target_outputs: List[np.ndarray]):
        """Train the SOM controller"""
        for input_vec, target_output in zip(input_data, target_outputs):
            bmu = self.find_bmu(input_vec)
            self.update_weights(input_vec, bmu)

            # Update output mapping
            self.output_map[bmu[0], bmu[1]] = target_output
```

## Learning-Based Control

### Operational Space Control

```python
import numpy as np
from scipy.spatial.transform import Rotation as R

class OperationalSpaceController:
    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.kp = 10.0  # Position gain
        self.kd = 2.0 * np.sqrt(self.kp)  # Critical damping
        self.kr = 10.0  # Orientation gain
        self.kw = 2.0 * np.sqrt(self.kr)  # Critical damping for orientation

    def compute_control(self, task_pose_desired: np.ndarray,
                       task_pose_current: np.ndarray,
                       task_vel_desired: np.ndarray,
                       task_vel_current: np.ndarray) -> np.ndarray:
        """Compute operational space control"""
        # Extract position and orientation
        pos_desired = task_pose_desired[:3]
        pos_current = task_pose_current[:3]
        rot_desired = task_pose_desired[3:]  # Quaternion
        rot_current = task_pose_current[3:]  # Quaternion

        # Position error
        pos_error = pos_desired - pos_current
        pos_vel_error = task_vel_desired[:3] - task_vel_current[:3]

        # Orientation error (using quaternion difference)
        rot_error = self._quaternion_error(rot_current, rot_desired)
        rot_vel_error = task_vel_desired[3:] - task_vel_current[3:]

        # Task space acceleration
        pos_acc = self.kp * pos_error + self.kd * pos_vel_error
        rot_acc = self.kr * rot_error + self.kw * rot_vel_error

        task_acc = np.concatenate([pos_acc, rot_acc])

        # Get Jacobian and its derivative
        jacobian = self.robot_model.get_jacobian()
        jacobian_dot = self.robot_model.get_jacobian_dot()

        # Mass matrix in task space
        mass_matrix = self.robot_model.get_mass_matrix()
        lambda_task = np.linalg.inv(jacobian @ np.linalg.inv(mass_matrix) @ jacobian.T)

        # Bias forces in task space
        bias_forces = self.robot_model.get_bias_forces()
        task_bias = jacobian @ np.linalg.inv(mass_matrix) @ bias_forces

        # Operational space control law
        joint_acc = (np.linalg.inv(mass_matrix) @
                    (jacobian.T @ (lambda_task @ task_acc + task_bias) -
                     self.robot_model.get_coriolis_matrix() @ task_vel_current))

        return joint_acc

    def _quaternion_error(self, q_current: np.ndarray, q_desired: np.ndarray) -> np.ndarray:
        """Compute orientation error using quaternions"""
        # q_error = q_desired * q_current^(-1)
        q_current_inv = np.array([q_current[0], -q_current[1], -q_current[2], -q_current[3]])
        q_error = self._quaternion_multiply(q_desired, q_current_inv)

        # Extract rotation vector from quaternion
        if q_error[0] > 0:  # Ensure shortest rotation
            q_error = -q_error

        # Convert to rotation vector
        angle = 2 * np.arccos(np.abs(q_error[0]))
        if angle < 1e-6:  # Very small rotation
            return np.zeros(3)

        s = np.sin(angle / 2)
        if abs(s) < 1e-6:
            return np.zeros(3)

        axis = q_error[1:] / s
        return axis * angle

    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2

        return np.array([w, x, y, z])
```

### Reinforcement Learning for Motor Control

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ActorNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action = self.tanh(self.fc3(x))  # Normalize to [-1, 1]
        return action

class CriticNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = self.fc3(x)
        return value

class DDPGAgent:
    def __init__(self, state_dim: int, action_dim: int, lr_actor: float = 1e-4,
                 lr_critic: float = 1e-3, tau: float = 0.005):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.actor = ActorNetwork(state_dim, action_dim).to(self.device)
        self.actor_target = ActorNetwork(state_dim, action_dim).to(self.device)
        self.critic = CriticNetwork(state_dim, action_dim).to(self.device)
        self.critic_target = CriticNetwork(state_dim, action_dim).to(self.device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Copy weights to target networks
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)

        self.tau = tau  # Soft update parameter

    def hard_update(self, target, source):
        """Hard update target network with source network weights"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def soft_update(self, target, source):
        """Soft update target network with source network weights"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def select_action(self, state, add_noise=True, noise_scale=0.1):
        """Select action using the actor network"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state_tensor).cpu().data.numpy()[0]

        if add_noise:
            noise = np.random.normal(0, noise_scale, size=action.shape)
            action = np.clip(action + noise, -1, 1)

        return action

    def update(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        """Update the DDPG networks"""
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).unsqueeze(1).to(self.device)

        # Update critic
        next_actions = self.actor_target(next_state_batch)
        next_q_values = self.critic_target(next_state_batch, next_actions)
        target_q_values = reward_batch + (0.99 * next_q_values * (1 - done_batch))

        current_q_values = self.critic(state_batch, action_batch)
        critic_loss = nn.MSELoss()(current_q_values, target_q_values)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        predicted_actions = self.actor(state_batch)
        actor_loss = -self.critic(state_batch, predicted_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)
```

## Multi-Modal Control Integration

### Force Control and Impedance Control

```python
class ImpedanceController:
    def __init__(self, mass: float = 1.0, damping: float = 10.0, stiffness: float = 100.0):
        self.mass = mass
        self.damping = damping
        self.stiffness = stiffness

        self.position_error = 0.0
        self.velocity_error = 0.0
        self.prev_time = None

    def update(self, desired_pos: float, current_pos: float,
               desired_vel: float = 0.0, current_vel: float = 0.0,
               dt: float = 0.01) -> float:
        """Update impedance controller"""
        # Calculate errors
        self.position_error = desired_pos - current_pos
        self.velocity_error = desired_vel - current_vel

        # Impedance control law: F = M*ẍ + B*ẋ + K*x
        force = (self.mass * (desired_vel - current_vel) / dt +
                self.damping * self.velocity_error +
                self.stiffness * self.position_error)

        return force

class HybridPositionForceController:
    def __init__(self, kp: float = 100.0, ki: float = 10.0, kd: float = 10.0,
                 force_threshold: float = 10.0):
        self.position_controller = PIDController(kp, ki, kd)
        self.force_controller = PIDController(kp/10, ki/10, kd/10)
        self.force_threshold = force_threshold

        self.contact_detected = False
        self.surface_normal = np.array([0.0, 0.0, 1.0])  # Assumed surface normal

    def update(self, position_setpoint: float, current_position: float,
               force_setpoint: float, current_force: float) -> float:
        """Update hybrid position/force controller"""
        # Detect contact based on force
        if abs(current_force) > self.force_threshold:
            self.contact_detected = True
        elif abs(current_force) < self.force_threshold * 0.5:
            self.contact_detected = False

        if self.contact_detected:
            # Force control mode
            force_error = force_setpoint - current_force
            position_correction = self.force_controller.update(0.0, force_error)
            desired_position = current_position + position_correction
        else:
            # Position control mode
            desired_position = position_setpoint

        # Position control
        position_error = desired_position - current_position
        control_output = self.position_controller.update(0.0, position_error)

        return control_output
```

## Real-Time Control Systems

### Control Loop Architecture

```python
import time
import threading
from collections import deque
import numpy as np

class RealTimeController:
    def __init__(self, control_frequency: float = 100.0):
        self.control_frequency = control_frequency
        self.dt = 1.0 / control_frequency
        self.running = False

        # Control components
        self.controllers = {}
        self.sensors = {}
        self.actuators = {}

        # Threading components
        self.control_thread = None
        self.sensor_thread = None

        # Data buffers
        self.sensor_buffer = deque(maxlen=10)
        self.command_buffer = deque(maxlen=10)

        # Performance monitoring
        self.loop_times = deque(maxlen=100)
        self.cpu_usage = 0.0

    def add_controller(self, name: str, controller):
        """Add a controller to the system"""
        self.controllers[name] = controller

    def add_sensor(self, name: str, sensor):
        """Add a sensor to the system"""
        self.sensors[name] = sensor

    def add_actuator(self, name: str, actuator):
        """Add an actuator to the system"""
        self.actuators[name] = actuator

    def start(self):
        """Start the real-time control system"""
        self.running = True

        # Start control thread
        self.control_thread = threading.Thread(target=self._control_loop)
        self.control_thread.start()

        # Start sensor thread
        self.sensor_thread = threading.Thread(target=self._sensor_loop)
        self.sensor_thread.start()

    def stop(self):
        """Stop the real-time control system"""
        self.running = False

        if self.control_thread:
            self.control_thread.join()

        if self.sensor_thread:
            self.sensor_thread.join()

    def _control_loop(self):
        """Main control loop running at specified frequency"""
        while self.running:
            start_time = time.time()

            # Read sensor data
            sensor_data = self._read_sensors()

            # Update controllers
            commands = self._update_controllers(sensor_data)

            # Send commands to actuators
            self._send_commands(commands)

            # Monitor performance
            loop_time = time.time() - start_time
            self.loop_times.append(loop_time)

            # Maintain control frequency
            sleep_time = max(0, self.dt - loop_time)
            time.sleep(sleep_time)

    def _sensor_loop(self):
        """Sensor reading loop (may run at different frequency)"""
        while self.running:
            sensor_data = self._read_sensors()
            self.sensor_buffer.append(sensor_data)
            time.sleep(0.001)  # High frequency sensor reading

    def _read_sensors(self) -> Dict:
        """Read all sensors"""
        data = {}
        for name, sensor in self.sensors.items():
            try:
                data[name] = sensor.read()
            except Exception as e:
                print(f"Error reading sensor {name}: {e}")
                data[name] = None
        return data

    def _update_controllers(self, sensor_data: Dict) -> Dict:
        """Update all controllers"""
        commands = {}
        for name, controller in self.controllers.items():
            try:
                command = controller.update(sensor_data)
                commands[name] = command
            except Exception as e:
                print(f"Error updating controller {name}: {e}")
                commands[name] = 0.0
        return commands

    def _send_commands(self, commands: Dict):
        """Send commands to actuators"""
        for name, command in commands.items():
            if name in self.actuators:
                try:
                    self.actuators[name].send_command(command)
                except Exception as e:
                    print(f"Error sending command to actuator {name}: {e}")

    def get_performance_metrics(self) -> Dict:
        """Get real-time performance metrics"""
        if self.loop_times:
            avg_loop_time = np.mean(self.loop_times)
            max_loop_time = max(self.loop_times)
            min_loop_time = min(self.loop_times)
            loop_time_std = np.std(self.loop_times)

            return {
                'avg_loop_time': avg_loop_time,
                'max_loop_time': max_loop_time,
                'min_loop_time': min_loop_time,
                'loop_time_std': loop_time_std,
                'frequency_achieved': 1.0 / avg_loop_time if avg_loop_time > 0 else 0,
                'jitter': loop_time_std
            }
        else:
            return {'avg_loop_time': 0, 'frequency_achieved': 0, 'jitter': 0}
```

## NVIDIA Isaac Control Integration

### Isaac ROS Control Components

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped
from control_msgs.msg import JointTrajectoryControllerState
import numpy as np

class IsaacControlNode(Node):
    def __init__(self):
        super().__init__('isaac_control_node')

        # Publishers and subscribers
        self.joint_command_pub = self.create_publisher(
            Float64MultiArray, '/joint_commands', 10)
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)
        self.wrench_sub = self.create_subscription(
            WrenchStamped, '/wrench', self.wrench_callback, 10)

        # Control components
        self.joint_controllers = {}
        self.trajectory_follower = TrajectoryFollower()
        self.impedance_controller = ImpedanceController()

        # Robot state
        self.current_positions = {}
        self.current_velocities = {}
        self.current_efforts = {}
        self.contact_force = np.zeros(3)

        # Control parameters
        self.declare_parameter('control_frequency', 100)
        self.control_frequency = self.get_parameter('control_frequency').value
        self.dt = 1.0 / self.control_frequency

        # Start control timer
        self.control_timer = self.create_timer(1.0/self.control_frequency, self.control_callback)

    def joint_state_callback(self, msg: JointState):
        """Update joint state from robot"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_positions[name] = msg.position[i]
            if i < len(msg.velocity):
                self.current_velocities[name] = msg.velocity[i]
            if i < len(msg.effort):
                self.current_efforts[name] = msg.effort[i]

    def wrench_callback(self, msg: WrenchStamped):
        """Update contact force from force sensor"""
        self.contact_force = np.array([
            msg.wrench.force.x,
            msg.wrench.force.y,
            msg.wrench.force.z
        ])

    def add_joint_controller(self, joint_name: str, kp: float, ki: float, kd: float):
        """Add a PID controller for a specific joint"""
        controller = PIDController(kp, ki, kd, self.dt)
        self.joint_controllers[joint_name] = controller

    def control_callback(self):
        """Main control callback running at control frequency"""
        commands = Float64MultiArray()

        # Update trajectory follower if active
        if self.trajectory_follower.trajectory:
            desired_pos, desired_vel, desired_acc = self.trajectory_follower.update()

            if not self.trajectory_follower.is_complete():
                # Apply impedance control
                force_command = self.impedance_controller.update(
                    desired_pos, self.current_positions.get('joint_1', 0.0),
                    desired_vel, self.current_velocities.get('joint_1', 0.0),
                    self.dt
                )

                # Convert force to joint torque (simplified)
                torque_command = force_command  # In practice, this would involve Jacobian

                commands.data.append(torque_command)

        # Publish joint commands
        self.joint_command_pub.publish(commands)

    def execute_trajectory(self, joint_names: List[str], start_pos: List[float],
                          end_pos: List[float], duration: float):
        """Execute a joint trajectory"""
        trajectory = MultiJointTrajectory(joint_names, start_pos, end_pos, duration)
        self.trajectory_follower.set_trajectory(trajectory)

    def enable_impedance_control(self, stiffness: float, damping: float):
        """Enable impedance control with specified parameters"""
        self.impedance_controller.stiffness = stiffness
        self.impedance_controller.damping = damping

    def get_robot_state(self) -> Dict:
        """Get current robot state"""
        return {
            'positions': dict(self.current_positions),
            'velocities': dict(self.current_velocities),
            'efforts': dict(self.current_efforts),
            'contact_force': self.contact_force.tolist()
        }
```

## Motor Skill Learning

### Dynamic Movement Primitives (DMP)

```python
class DMP:
    def __init__(self, n_basis: int = 10, alpha: float = 49.0):
        self.n_basis = n_basis
        self.alpha = alpha
        self.beta = alpha / 4.0  # Critical damping
        self.alpha_z = 3.0

        # Basis functions parameters
        self.centers = np.linspace(0, 1, n_basis)
        self.widths = np.ones(n_basis) * n_basis / 1.5

        # Weights (to be learned)
        self.weights = np.zeros(n_basis)

        # Initial and goal states
        self.y0 = 0.0
        self.goal = 1.0
        self.phase = 1.0  # Start with phase variable at 1

    def set_goal(self, y0: float, goal: float):
        """Set initial and goal states"""
        self.y0 = y0
        self.goal = goal

    def train(self, trajectory: np.ndarray, time_steps: np.ndarray):
        """Train DMP to reproduce a given trajectory"""
        # Calculate phase variable
        s = np.exp(-self.alpha_z * time_steps / time_steps[-1])

        # Calculate desired velocity and acceleration
        dy_desired = np.gradient(trajectory) / (time_steps[1] - time_steps[0])
        ddy_desired = np.gradient(dy_desired) / (time_steps[1] - time_steps[0])

        # Calculate forcing term
        forcing_term = (ddy_desired - self.alpha * (self.beta * (self.goal - trajectory) - dy_desired))

        # Calculate basis function activations
        basis_activations = np.zeros((len(time_steps), self.n_basis))
        for i, t in enumerate(time_steps):
            for j in range(self.n_basis):
                basis_activations[i, j] = (np.exp(-self.widths[j] * (s[i] - self.centers[j])**2) *
                                        s[i] * (self.goal - self.y0))

        # Learn weights using least squares
        if np.linalg.matrix_rank(basis_activations) == basis_activations.shape[1]:
            self.weights = np.linalg.lstsq(basis_activations, forcing_term, rcond=None)[0]
        else:
            # Use pseudo-inverse for rank-deficient case
            self.weights = np.linalg.pinv(basis_activations) @ forcing_term

    def step(self, dt: float, y: float, dy: float) -> Tuple[float, float]:
        """Perform one step of DMP integration"""
        # Update phase variable
        ds = -self.alpha_z * self.phase * dt
        self.phase += ds

        # Calculate basis function activations
        basis_values = np.exp(-self.widths * (self.phase - self.centers)**2) * self.phase

        # Calculate forcing term
        forcing = np.dot(self.weights, basis_values) * self.phase

        # DMP equation
        ddy = (self.alpha * (self.beta * (self.goal - y) - dy) + forcing) * self.phase
        dy_new = dy + ddy * dt
        y_new = y + dy_new * dt

        return y_new, dy_new

    def generate_trajectory(self, duration: float, dt: float = 0.01) -> np.ndarray:
        """Generate trajectory using learned DMP"""
        steps = int(duration / dt)
        trajectory = np.zeros(steps)
        velocities = np.zeros(steps)

        y, dy = self.y0, 0.0

        for i in range(steps):
            y, dy = self.step(dt, y, dy)
            trajectory[i] = y
            velocities[i] = dy

        return trajectory, velocities
```

## Control Quality Assessment

### Control Performance Metrics

```python
class ControlQualityAssessor:
    def __init__(self):
        self.tracking_errors = []
        self.control_efforts = []
        self.stability_metrics = []

    def assess_tracking_performance(self, desired_trajectory: np.ndarray,
                                  actual_trajectory: np.ndarray) -> Dict[str, float]:
        """Assess tracking performance of control system"""
        if len(desired_trajectory) != len(actual_trajectory):
            raise ValueError("Trajectories must have same length")

        # Calculate tracking error
        errors = desired_trajectory - actual_trajectory
        self.tracking_errors.extend(errors)

        # Performance metrics
        rmse = np.sqrt(np.mean(errors**2))
        max_error = np.max(np.abs(errors))
        mean_error = np.mean(np.abs(errors))

        # Integral of absolute error
        iae = np.sum(np.abs(errors)) * 0.01  # Assuming dt = 0.01

        # Integral of squared error
        ise = np.sum(errors**2) * 0.01

        # Calculate settling time (time to stay within 2% of final value)
        final_value = desired_trajectory[-1]
        threshold = 0.02 * abs(final_value) if abs(final_value) > 1e-6 else 0.02
        settling_idx = -1
        for i in range(len(errors)-1, -1, -1):
            if abs(errors[i]) > threshold:
                settling_idx = i
                break

        settling_time = len(errors) * 0.01 if settling_idx == -1 else settling_idx * 0.01

        return {
            'rmse': rmse,
            'max_error': max_error,
            'mean_error': mean_error,
            'iae': iae,
            'ise': ise,
            'settling_time': settling_time,
            'overshoot': self._calculate_overshoot(desired_trajectory, actual_trajectory)
        }

    def _calculate_overshoot(self, desired: np.ndarray, actual: np.ndarray) -> float:
        """Calculate percentage overshoot"""
        final_desired = desired[-1]
        max_actual = np.max(actual)
        min_actual = np.min(actual)

        if final_desired > 0:
            overshoot = max(0, (max_actual - final_desired) / final_desired) * 100
        else:
            overshoot = max(0, (final_desired - min_actual) / abs(final_desired)) * 100

        return overshoot

    def assess_stability(self, control_signals: np.ndarray) -> Dict[str, float]:
        """Assess stability of control system"""
        # Calculate control effort
        control_effort = np.mean(np.abs(control_signals))
        self.control_efforts.append(control_effort)

        # Check for oscillations
        zero_crossings = self._count_zero_crossings(control_signals)
        oscillation_frequency = zero_crossings / len(control_signals) * 100  # Hz

        # Check for chattering (high frequency oscillations)
        gradient = np.gradient(control_signals)
        chattering = np.mean(np.abs(gradient))

        # Frequency domain analysis
        fft_signal = np.fft.fft(control_signals)
        power_spectrum = np.abs(fft_signal)**2
        dominant_freq_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
        dominant_frequency = dominant_freq_idx * 100 / len(control_signals)  # Assuming 100 Hz sampling

        stability_score = self._calculate_stability_score(control_signals)

        return {
            'control_effort': control_effort,
            'zero_crossings': zero_crossings,
            'oscillation_frequency': oscillation_frequency,
            'chattering': chattering,
            'dominant_frequency': dominant_frequency,
            'stability_score': stability_score
        }

    def _count_zero_crossings(self, signal: np.ndarray) -> int:
        """Count zero crossings in signal"""
        return np.sum(np.diff(np.sign(signal)) != 0)

    def _calculate_stability_score(self, signal: np.ndarray) -> float:
        """Calculate stability score based on signal characteristics"""
        # Variance-based stability measure
        variance = np.var(signal)
        stability = 1.0 / (1.0 + variance)

        # Frequency content stability measure
        fft_signal = np.abs(np.fft.fft(signal))
        high_freq_content = np.mean(fft_signal[len(fft_signal)//2:])

        # Combine metrics
        score = stability * (1.0 / (1.0 + high_freq_content))

        return min(score, 1.0)  # Clamp to [0, 1]

    def detect_control_anomalies(self, current_metrics: Dict) -> List[str]:
        """Detect anomalies in control performance"""
        anomalies = []

        # Check for sudden increases in tracking error
        if len(self.tracking_errors) >= 100:
            recent_errors = self.tracking_errors[-50:]
            historical_errors = self.tracking_errors[:-50]

            if historical_errors:
                recent_rmse = np.sqrt(np.mean(np.array(recent_errors)**2))
                historical_rmse = np.sqrt(np.mean(np.array(historical_errors)**2))

                if recent_rmse > 2 * historical_rmse:
                    anomalies.append('tracking_error_spike')

        # Check for excessive control effort
        if len(self.control_efforts) >= 10:
            recent_effort = np.mean(self.control_efforts[-5:])
            historical_effort = np.mean(self.control_efforts[:-5])

            if recent_effort > 1.5 * historical_effort:
                anomalies.append('excessive_control_effort')

        return anomalies
```

## Key Takeaways

- Control systems translate high-level plans into precise physical actions
- PID controllers provide fundamental feedback control capabilities
- Trajectory generation ensures smooth, controlled motion
- Adaptive control systems adjust to changing conditions and uncertainties
- Learning-based control enables skill acquisition and improvement
- Real-time control systems require careful timing and performance optimization
- Quality assessment ensures reliable and stable control performance

## Next Steps

In the next chapter, we'll explore learning and adaptation mechanisms that enable AI robot brains to improve their performance over time through experience and interaction with the environment.