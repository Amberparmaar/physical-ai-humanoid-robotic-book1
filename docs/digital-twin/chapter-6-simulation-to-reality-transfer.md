---
sidebar_position: 6
title: Simulation-to-Reality Transfer
---

# Simulation-to-Reality Transfer

In this chapter, we'll explore the critical challenge of transferring knowledge and behaviors from simulation to the real world. This "sim-to-real" transfer is essential for robotics development, as it allows us to train and test algorithms in safe, controlled simulation environments before deploying them on physical robots.

## Understanding the Reality Gap

The "reality gap" refers to the differences between simulated and real environments that can cause algorithms trained in simulation to fail when deployed on real robots. These differences include:

### Physical Differences
- **Dynamics**: Friction, compliance, and other physical properties may not be perfectly modeled
- **Actuator behavior**: Real motors have delays, noise, and limitations not captured in simulation
- **Sensor noise**: Real sensors have different noise patterns and characteristics than simulated ones
- **Environmental conditions**: Lighting, temperature, and other environmental factors vary

### Perception Differences
- **Visual appearance**: Colors, textures, and lighting differ between simulation and reality
- **Sensor limitations**: Real sensors have different resolution, range, and accuracy
- **Occlusions**: Real environments have more complex occlusion patterns

### Behavioral Differences
- **Timing**: Real systems have communication delays and processing latencies
- **Uncertainty**: Real environments have more unpredictable elements
- **Wear and tear**: Physical components degrade over time

## Domain Randomization

Domain randomization is a technique that increases the diversity of simulated environments to improve sim-to-real transfer:

### Visual Domain Randomization

```python
import random
import numpy as np
from PIL import Image, ImageEnhance

class VisualDomainRandomizer:
    def __init__(self):
        self.lighting_conditions = [
            {"brightness": (0.7, 1.3), "contrast": (0.8, 1.2), "saturation": (0.8, 1.2)},
            {"brightness": (0.5, 1.5), "contrast": (0.6, 1.4), "saturation": (0.6, 1.4)},
            {"brightness": (0.9, 1.1), "contrast": (0.9, 1.1), "saturation": (0.9, 1.1)}
        ]

        self.textures = [
            "wood", "metal", "concrete", "carpet", "tile", "grass", "sand"
        ]

        self.camera_params = {
            "noise_std": (0.001, 0.01),
            "blur_radius": (0.1, 2.0),
            "color_shift": (0.0, 0.1)
        }

    def randomize_image(self, image):
        """Apply random visual transformations to an image"""
        img = Image.fromarray(image)

        # Random brightness adjustment
        brightness_factor = random.uniform(*self.lighting_conditions[0]["brightness"])
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness_factor)

        # Random contrast adjustment
        contrast_factor = random.uniform(*self.lighting_conditions[0]["contrast"])
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast_factor)

        # Random saturation adjustment
        saturation_factor = random.uniform(*self.lighting_conditions[0]["saturation"])
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(saturation_factor)

        # Add random noise
        noise_std = random.uniform(*self.camera_params["noise_std"])
        img_array = np.array(img).astype(np.float32)
        noise = np.random.normal(0, noise_std, img_array.shape)
        img_array = np.clip(img_array + noise, 0, 255)
        img = Image.fromarray(img_array.astype(np.uint8))

        return np.array(img)

    def randomize_environment(self, sim_env):
        """Randomize environment properties"""
        # Randomize lighting
        light_intensity = random.uniform(0.5, 2.0)
        sim_env.set_light_intensity(light_intensity)

        # Randomize textures
        random_texture = random.choice(self.textures)
        sim_env.set_floor_texture(random_texture)

        # Randomize object properties
        for obj in sim_env.get_objects():
            if random.random() < 0.3:  # 30% chance to modify object
                color = (random.random(), random.random(), random.random())
                obj.set_color(color)

        return sim_env
```

### Physical Domain Randomization

```python
import random

class PhysicalDomainRandomizer:
    def __init__(self):
        # Range of physical parameters to randomize
        self.params = {
            "friction": (0.1, 1.0),
            "mass_multiplier": (0.8, 1.2),
            "inertia_multiplier": (0.8, 1.2),
            "motor_delay": (0.01, 0.05),
            "sensor_noise": (0.001, 0.01),
            "gravity": (9.5, 10.1)  # Slight variations in gravity
        }

    def randomize_robot(self, robot):
        """Apply random physical properties to robot"""
        # Randomize link masses
        for link in robot.links:
            mass_factor = random.uniform(*self.params["mass_multiplier"])
            link.mass *= mass_factor

            # Randomize friction
            friction = random.uniform(*self.params["friction"])
            link.set_friction(friction)

        # Randomize joint properties
        for joint in robot.joints:
            if joint.type in ["revolute", "continuous"]:
                damping_factor = random.uniform(0.8, 1.2)
                joint.damping *= damping_factor

        # Add random delays to motor responses
        robot.motor_delay = random.uniform(*self.params["motor_delay"])

        return robot

    def randomize_environment(self, env):
        """Randomize environment physics"""
        # Randomize gravity
        gravity_factor = random.uniform(*self.params["gravity"])
        env.set_gravity([0, 0, -gravity_factor])

        # Randomize surface properties
        for surface in env.get_surfaces():
            friction = random.uniform(*self.params["friction"])
            surface.set_friction(friction)

        return env
```

## System Identification and Parameter Tuning

### Parameter Estimation

```python
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class SystemIdentifier:
    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.sim_params = {}
        self.real_params = {}

    def collect_data(self, robot, inputs, duration=10.0):
        """Collect input-output data from robot"""
        data = {
            'time': [],
            'inputs': [],
            'outputs': [],
            'states': []
        }

        dt = 0.01  # 100 Hz
        for t in np.arange(0, duration, dt):
            # Apply input
            robot.apply_control(inputs[int(t/dt)])

            # Record state
            state = robot.get_state()
            output = robot.get_sensor_data()

            data['time'].append(t)
            data['inputs'].append(inputs[int(t/dt)])
            data['outputs'].append(output)
            data['states'].append(state)

        return data

    def identify_parameters(self, sim_data, real_data):
        """Identify parameters that minimize sim-to-real discrepancy"""
        def objective_function(params):
            # Update simulation with new parameters
            self.update_simulation_params(params)

            # Run simulation with same inputs as real data
            sim_output = self.run_simulation_with_params(
                real_data['inputs'], params
            )

            # Calculate difference between sim and real
            diff = np.array(sim_output) - np.array(real_data['outputs'])
            error = np.mean(np.square(diff))
            return error

        # Initial parameter guess
        initial_params = self.get_initial_params()

        # Optimize parameters
        result = minimize(
            objective_function,
            initial_params,
            method='BFGS',
            options={'disp': True}
        )

        return result.x

    def update_simulation_params(self, params):
        """Update simulation with identified parameters"""
        # Update robot model with new parameters
        for i, param_name in enumerate(self.get_parameter_names()):
            self.robot_model.set_parameter(param_name, params[i])

    def get_initial_params(self):
        """Get initial parameter values"""
        param_names = self.get_parameter_names()
        return [self.robot_model.get_parameter(name) for name in param_names]

    def get_parameter_names(self):
        """Get list of parameters to identify"""
        return [
            'link_mass_1', 'link_mass_2', 'friction_1', 'friction_2',
            'inertia_1', 'inertia_2', 'motor_constant', 'sensor_noise'
        ]
```

## Domain Adaptation Techniques

### Adversarial Domain Adaptation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DomainAdaptationNetwork(nn.Module):
    def __init__(self, input_dim, feature_dim=256):
        super(DomainAdaptationNetwork, self).__init__()

        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim),
            nn.ReLU()
        )

        # Task classifier
        self.task_classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)  # Example: 10 classes
        )

        # Domain classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # 2 domains: sim and real
        )

    def forward(self, x, alpha=0.0):
        # Extract features
        features = self.feature_extractor(x)

        # Reverse gradient for domain adaptation
        reverse_features = ReverseGradient.apply(features, alpha)

        # Task prediction
        task_output = self.task_classifier(features)

        # Domain prediction
        domain_output = self.domain_classifier(reverse_features)

        return task_output, domain_output

class ReverseGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, alpha):
        ctx.alpha = alpha
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return ctx.alpha * grad_output.neg(), None

# Training loop for domain adaptation
def train_domain_adaptation(model, sim_loader, real_loader, num_epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    task_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        for (sim_data, sim_labels), (real_data, _) in zip(sim_loader, real_loader):
            optimizer.zero_grad()

            # Combine data
            all_data = torch.cat([sim_data, real_data], dim=0)
            domain_labels = torch.cat([
                torch.zeros(sim_data.size(0)),  # Sim domain = 0
                torch.ones(real_data.size(0))   # Real domain = 1
            ]).long()

            # Train with gradually increasing domain confusion
            p = epoch / num_epochs
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            task_output, domain_output = model(all_data, alpha)

            # Task loss (only on simulated data with labels)
            task_loss = task_criterion(task_output[:sim_data.size(0)], sim_labels)

            # Domain loss (try to confuse domain classifier)
            domain_loss = domain_criterion(domain_output, domain_labels)

            total_loss = task_loss + domain_loss
            total_loss.backward()
            optimizer.step()
```

## Robust Control Design

### Robust Controller with Uncertainty

```python
import numpy as np
from scipy import signal
import control  # python-control package

class RobustController:
    def __init__(self, nominal_model, uncertainty_bounds):
        self.nominal_model = nominal_model
        self.uncertainty_bounds = uncertainty_bounds

    def design_robust_controller(self):
        """Design a robust controller using H-infinity methods"""
        # Define weighting functions for robustness
        W1 = self.create_weighting_function(
            low_freq_gain=1.0,
            high_freq_gain=0.1,
            corner_freq=1.0
        )

        W2 = self.create_weighting_function(
            low_freq_gain=0.1,
            high_freq_gain=1.0,
            corner_freq=10.0
        )

        # Formulate the mixed sensitivity problem
        P = self.create_augmented_plant(
            self.nominal_model, W1, W2
        )

        # Synthesize H-infinity controller
        K, cl, info = control.mixsyn(P, 1, 1, 1)

        return K

    def create_weighting_function(self, low_freq_gain, high_freq_gain, corner_freq):
        """Create a first-order weighting function"""
        # W(s) = (s/w1 + 1) / (s/w2 + 1) * gain
        w1 = corner_freq * low_freq_gain
        w2 = corner_freq * high_freq_gain
        gain = np.sqrt(low_freq_gain * high_freq_gain)

        num = [1/w2, gain]
        den = [1/w1, 1]

        return signal.TransferFunction(num, den)

    def create_augmented_plant(self, P, W1, W2):
        """Create augmented plant for mixed sensitivity"""
        # This is a simplified representation
        # In practice, this involves creating a larger system
        # that includes the weighting functions

        # Return augmented plant (simplified)
        return control.append(P, W1, W2)

class AdaptiveController:
    def __init__(self, initial_params):
        self.params = initial_params
        self.learning_rate = 0.01

    def update_parameters(self, error, state, input_signal):
        """Update controller parameters based on tracking error"""
        # Gradient-based parameter update
        param_gradients = self.compute_parameter_gradients(error, state, input_signal)

        for i, param in enumerate(self.params):
            self.params[i] -= self.learning_rate * param_gradients[i]

    def compute_parameter_gradients(self, error, state, input_signal):
        """Compute gradients of error w.r.t. parameters"""
        # Simplified gradient computation
        # In practice, this would use more sophisticated methods
        gradients = np.zeros_like(self.params)

        # Example: update based on state error
        gradients[0] += error * state[0]  # Proportional term
        gradients[1] += error * input_signal  # Integral term

        return gradients
```

## Sim-to-Real Transfer Strategies

### Progressive Domain Transfer

```python
class ProgressiveDomainTransfer:
    def __init__(self, base_env, real_env):
        self.base_env = base_env
        self.real_env = real_env
        self.transfer_stage = 0
        self.max_stages = 5

    def train_progressive(self, policy, num_iterations=1000):
        """Train policy with progressive domain transfer"""
        for stage in range(self.max_stages):
            self.transfer_stage = stage

            # Interpolate between sim and real environments
            env = self.interpolate_environments(stage)

            # Train policy in current environment
            policy = self.train_in_environment(policy, env, num_iterations)

            # Evaluate on real environment
            real_performance = self.evaluate_on_real(policy)

            print(f"Stage {stage}: Real performance = {real_performance}")

            if real_performance > 0.9:  # Threshold for success
                break

        return policy

    def interpolate_environments(self, stage):
        """Create environment that interpolates between sim and real"""
        # Calculate interpolation factor
        alpha = stage / (self.max_stages - 1)

        # Interpolate environment parameters
        interp_env = self.base_env.copy()

        # Gradually introduce real-world characteristics
        interp_env.friction = self.interpolate_param(
            self.base_env.friction,
            self.real_env.friction,
            alpha
        )

        interp_env.noise_level = self.interpolate_param(
            self.base_env.noise_level,
            self.real_env.noise_level,
            alpha
        )

        # Add sensor noise that matches real sensors
        if alpha > 0.5:
            interp_env.add_sensor_noise(self.real_env.get_sensor_noise_profile())

        return interp_env

    def interpolate_param(self, sim_val, real_val, alpha):
        """Linear interpolation between sim and real values"""
        return sim_val * (1 - alpha) + real_val * alpha

    def train_in_environment(self, policy, env, iterations):
        """Train policy in given environment"""
        for i in range(iterations):
            state = env.reset()
            done = False

            while not done:
                action = policy.get_action(state)
                next_state, reward, done, info = env.step(action)
                policy.update(state, action, reward, next_state, done)
                state = next_state

        return policy

    def evaluate_on_real(self, policy):
        """Evaluate policy performance on real environment"""
        total_reward = 0
        num_episodes = 10

        for _ in range(num_episodes):
            state = self.real_env.reset()
            episode_reward = 0
            done = False

            while not done:
                action = policy.get_action(state)
                state, reward, done, _ = self.real_env.step(action)
                episode_reward += reward

            total_reward += episode_reward

        return total_reward / num_episodes
```

## Reality Check and Validation

### Sim-to-Real Validation Framework

```python
class SimToRealValidator:
    def __init__(self, sim_model, real_robot):
        self.sim_model = sim_model
        self.real_robot = real_robot
        self.metrics = {}

    def validate_behavior(self, behavior_function):
        """Validate that behavior works in both sim and real"""
        # Test in simulation
        sim_result = self.test_in_simulation(behavior_function)

        # Test on real robot
        real_result = self.test_on_real_robot(behavior_function)

        # Compare results
        similarity_score = self.calculate_similarity(sim_result, real_result)

        self.metrics['behavior_similarity'] = similarity_score
        self.metrics['sim_performance'] = sim_result['performance']
        self.metrics['real_performance'] = real_result['performance']

        return similarity_score > 0.8  # Threshold for acceptable transfer

    def test_in_simulation(self, behavior_function):
        """Test behavior in simulation"""
        # Reset simulation to initial state
        self.sim_model.reset()

        # Execute behavior and collect metrics
        initial_state = self.sim_model.get_state()
        behavior_result = behavior_function(self.sim_model)
        final_state = self.sim_model.get_state()

        performance = self.calculate_performance(
            initial_state, final_state, behavior_result
        )

        return {
            'performance': performance,
            'trajectory': self.sim_model.get_trajectory(),
            'time': self.sim_model.get_execution_time()
        }

    def test_on_real_robot(self, behavior_function):
        """Test behavior on real robot"""
        # Reset real robot to initial state
        self.real_robot.reset()

        # Execute behavior and collect metrics
        initial_state = self.real_robot.get_state()
        behavior_result = behavior_function(self.real_robot)
        final_state = self.real_robot.get_state()

        performance = self.calculate_performance(
            initial_state, final_state, behavior_result
        )

        return {
            'performance': performance,
            'trajectory': self.real_robot.get_trajectory(),
            'time': self.real_robot.get_execution_time()
        }

    def calculate_similarity(self, sim_data, real_data):
        """Calculate similarity between sim and real results"""
        # Compare trajectories
        trajectory_similarity = self.compare_trajectories(
            sim_data['trajectory'], real_data['trajectory']
        )

        # Compare performance metrics
        performance_similarity = abs(
            sim_data['performance'] - real_data['performance']
        ) / max(sim_data['performance'], real_data['performance'])

        # Compare execution times
        time_similarity = abs(
            sim_data['time'] - real_data['time']
        ) / max(sim_data['time'], real_data['time'])

        # Weighted average of similarities
        total_similarity = (
            0.5 * trajectory_similarity +
            0.3 * (1 - performance_similarity) +
            0.2 * (1 - time_similarity)
        )

        return total_similarity

    def compare_trajectories(self, traj1, traj2):
        """Compare two trajectories for similarity"""
        if len(traj1) == 0 or len(traj2) == 0:
            return 0.0

        # Normalize trajectories to same length
        min_len = min(len(traj1), len(traj2))
        traj1_norm = traj1[:min_len]
        traj2_norm = traj2[:min_len]

        # Calculate average distance between trajectories
        distances = []
        for p1, p2 in zip(traj1_norm, traj2_norm):
            dist = np.linalg.norm(np.array(p1) - np.array(p2))
            distances.append(dist)

        avg_distance = np.mean(distances)

        # Convert to similarity (0-1 scale, where 1 is identical)
        max_expected_distance = 1.0  # Adjust based on your application
        similarity = max(0, 1 - avg_distance / max_expected_distance)

        return similarity

    def calculate_performance(self, initial_state, final_state, behavior_result):
        """Calculate performance metric for behavior"""
        # Example: distance to goal, energy efficiency, etc.
        # This should be customized based on your specific task

        # Placeholder implementation
        return float(behavior_result.get('success', 0))
```

## Practical Transfer Examples

### Reinforcement Learning Transfer

```python
import torch
import torch.nn as nn
import numpy as np

class SimToRealRLTransfer:
    def __init__(self, sim_env, real_env, policy_network):
        self.sim_env = sim_env
        self.real_env = real_env
        self.policy_network = policy_network
        self.sim_buffer = []
        self.real_buffer = []

    def train_with_domain_randomization(self, episodes=10000):
        """Train policy with domain randomization"""
        for episode in range(episodes):
            # Randomize simulation environment
            self.randomize_simulation()

            # Collect experience in simulation
            sim_experience = self.collect_experience(self.sim_env)
            self.sim_buffer.extend(sim_experience)

            # Update policy using simulation data
            self.update_policy(self.sim_buffer)

            # Occasionally test on real robot
            if episode % 1000 == 0:
                real_experience = self.collect_experience(self.real_env)
                self.real_buffer.extend(real_experience)

                # Fine-tune with real data
                if len(self.real_buffer) > 100:
                    self.finetune_with_real_data()

    def randomize_simulation(self):
        """Apply domain randomization to simulation"""
        # Randomize physical parameters
        friction = np.random.uniform(0.1, 1.0)
        mass_variation = np.random.uniform(0.8, 1.2)

        self.sim_env.set_friction(friction)
        self.sim_env.set_mass_variation(mass_variation)

        # Randomize visual appearance
        lighting = np.random.uniform(0.5, 2.0)
        self.sim_env.set_lighting(lighting)

        # Randomize sensor noise
        noise_level = np.random.uniform(0.001, 0.01)
        self.sim_env.set_sensor_noise(noise_level)

    def collect_experience(self, env):
        """Collect experience from environment"""
        experience = []
        state = env.reset()
        done = False

        while not done:
            action = self.get_action(state)
            next_state, reward, done, info = env.step(action)

            experience.append({
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': done
            })

            state = next_state

        return experience

    def update_policy(self, buffer):
        """Update policy using experience buffer"""
        if len(buffer) < 32:  # Batch size
            return

        # Sample batch
        batch = np.random.choice(buffer, 32)

        states = torch.FloatTensor([exp['state'] for exp in batch])
        actions = torch.LongTensor([exp['action'] for exp in batch])
        rewards = torch.FloatTensor([exp['reward'] for exp in batch])
        next_states = torch.FloatTensor([exp['next_state'] for exp in batch])
        dones = torch.BoolTensor([exp['done'] for exp in batch])

        # Compute loss and update
        current_q = self.policy_network(states).gather(1, actions.unsqueeze(1))
        next_q = self.policy_network(next_states).max(1)[0].detach()
        target_q = rewards + (0.99 * next_q * ~dones)

        loss = nn.MSELoss()(current_q.squeeze(), target_q)

        # Backpropagate
        self.policy_network.optimizer.zero_grad()
        loss.backward()
        self.policy_network.optimizer.step()

    def finetune_with_real_data(self):
        """Fine-tune policy with real robot data"""
        # Use a smaller learning rate for real data
        original_lr = self.policy_network.optimizer.param_groups[0]['lr']
        self.policy_network.optimizer.param_groups[0]['lr'] = original_lr * 0.1

        # Train on real data
        self.update_policy(self.real_buffer[-100:])  # Use last 100 real experiences

        # Restore original learning rate
        self.policy_network.optimizer.param_groups[0]['lr'] = original_lr
```

## Transfer Success Metrics

### Comprehensive Evaluation Framework

```python
class TransferEvaluator:
    def __init__(self):
        self.metrics = {}

    def evaluate_transfer_success(self, sim_policy, real_policy, test_scenarios):
        """Evaluate transfer success across multiple metrics"""
        results = {
            'performance_preservation': [],
            'behavioral_similarity': [],
            'robustness': [],
            'generalization': []
        }

        for scenario in test_scenarios:
            # Performance preservation
            sim_perf = self.evaluate_policy(sim_policy, scenario, 'sim')
            real_perf = self.evaluate_policy(real_policy, scenario, 'real')
            perf_pres = self.calculate_performance_preservation(sim_perf, real_perf)
            results['performance_preservation'].append(perf_pres)

            # Behavioral similarity
            sim_traj = self.get_trajectory(sim_policy, scenario, 'sim')
            real_traj = self.get_trajectory(real_policy, scenario, 'real')
            beh_sim = self.calculate_behavioral_similarity(sim_traj, real_traj)
            results['behavioral_similarity'].append(beh_sim)

            # Robustness (variance across different conditions)
            robustness = self.evaluate_robustness(real_policy, scenario)
            results['robustness'].append(robustness)

            # Generalization (performance on unseen scenarios)
            gen_score = self.evaluate_generalization(real_policy, scenario)
            results['generalization'].append(gen_score)

        # Calculate overall transfer score
        overall_score = np.mean([
            np.mean(results['performance_preservation']),
            np.mean(results['behavioral_similarity']),
            np.mean(results['robustness']),
            np.mean(results['generalization'])
        ])

        return {
            'overall_score': overall_score,
            'detailed_results': results,
            'transfer_success': overall_score > 0.7  # Threshold
        }

    def evaluate_policy(self, policy, scenario, domain):
        """Evaluate policy performance in given scenario"""
        # Implementation depends on specific task
        # Return performance metric (0-1 scale)
        pass

    def calculate_performance_preservation(self, sim_perf, real_perf):
        """Calculate how well performance was preserved"""
        if sim_perf == 0:
            return 0.0
        return max(0, real_perf / sim_perf)  # Clipped to [0,1]

    def calculate_behavioral_similarity(self, sim_traj, real_traj):
        """Calculate similarity between trajectories"""
        # Use Dynamic Time Warping or other trajectory similarity measures
        pass

    def evaluate_robustness(self, policy, scenario):
        """Evaluate policy robustness to perturbations"""
        base_performance = self.evaluate_policy(policy, scenario, 'real')

        perturbations = [
            {'noise_level': 0.01},
            {'friction': 1.2},
            {'mass': 1.1}
        ]

        perturbed_performances = []
        for perturb in perturbations:
            perturbed_perf = self.evaluate_policy_with_perturbation(
                policy, scenario, perturb
            )
            perturbed_performances.append(perturbed_perf)

        # Robustness = average performance under perturbations
        return np.mean(perturbed_performances) / base_performance

    def evaluate_generalization(self, policy, scenario):
        """Evaluate policy generalization to new scenarios"""
        # Test on scenarios not seen during training
        # Return generalization score
        pass
```

## Key Takeaways

- Domain randomization helps improve sim-to-real transfer by increasing simulation diversity
- System identification can bridge the reality gap by tuning simulation parameters
- Domain adaptation techniques help align simulation and reality representations
- Progressive transfer strategies gradually introduce real-world characteristics
- Robust control design accounts for model uncertainties
- Comprehensive validation ensures successful transfer

## Next Steps

In the final chapter of this module, we'll explore advanced applications and case studies that demonstrate state-of-the-art digital twin implementations in robotics, bringing together all the concepts we've learned.