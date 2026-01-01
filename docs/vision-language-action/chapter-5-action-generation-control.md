---
sidebar_position: 5
title: Action Generation and Control
---

# Action Generation and Control

In this chapter, we'll explore how Vision-Language-Action (VLA) systems generate and execute physical actions based on integrated vision and language understanding. Action generation is the culmination of perception and cognition, translating high-level goals into executable motor commands that allow robots to interact with the physical world.

## Understanding Action Generation in VLA Systems

Action generation in VLA systems involves several key components:

1. **Goal Interpretation**: Understanding what needs to be accomplished from language instructions
2. **Perception Integration**: Incorporating visual information to understand the current state
3. **Action Planning**: Generating sequences of actions to achieve the goal
4. **Motor Control**: Executing precise movements to carry out actions
5. **Feedback Integration**: Monitoring execution and adapting to changes

### The Action Generation Pipeline

The VLA action generation pipeline typically follows this sequence:

1. **Language Understanding**: Parse the instruction to extract goals and constraints
2. **Scene Understanding**: Analyze the visual scene to identify relevant objects and affordances
3. **Action Selection**: Choose appropriate actions based on the goal and scene
4. **Trajectory Planning**: Generate detailed movement trajectories
5. **Motor Execution**: Execute the planned actions with appropriate control
6. **Monitoring and Adaptation**: Monitor execution and adapt as needed

## Advanced Action Generation Architectures

### Hierarchical Action Generation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
import math

class HierarchicalActionGenerator(nn.Module):
    def __init__(self, d_model: int = 768, action_space_dim: int = 6, max_subtasks: int = 10):
        super().__init__()
        self.d_model = d_model
        self.action_space_dim = action_space_dim
        self.max_subtasks = max_subtasks

        # High-level goal interpreter
        self.goal_interpreter = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )

        # Subtask generator
        self.subtask_generator = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, max_subtasks * 2),  # 2 tokens per subtask (start, end)
            nn.Softmax(dim=-1)
        )

        # Low-level action generator
        self.low_level_generator = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # Combined goal and subtask features
            nn.ReLU(),
            nn.Linear(d_model, action_space_dim)
        )

        # Action parameter predictor
        self.param_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 20)  # Predict 20 action parameters
        )

    def forward(self, goal_features, scene_features, language_features):
        """
        Generate hierarchical actions from goal, scene, and language features
        Args:
            goal_features: Goal-specific features (batch_size, d_model)
            scene_features: Scene understanding features (batch_size, num_objects, d_model)
            language_features: Language instruction features (batch_size, seq_len, d_model)
        """
        batch_size = goal_features.size(0)

        # Interpret the high-level goal
        interpreted_goal = self.goal_interpreter(goal_features)

        # Generate subtasks based on goal and scene
        combined_features = torch.cat([
            interpreted_goal.unsqueeze(1).expand(-1, scene_features.size(1), -1),
            scene_features
        ], dim=-1)

        subtask_attention = self.subtask_generator(combined_features.mean(dim=1))  # (batch_size, max_subtasks * 2)

        # Reshape to get subtask start/end probabilities
        subtask_probs = subtask_attention.view(batch_size, self.max_subtasks, 2)

        # Generate low-level actions for each subtask
        low_level_actions = []
        action_params = []

        for i in range(self.max_subtasks):
            # Combine goal, scene, and subtask-specific features
            subtask_specific_features = torch.cat([
                interpreted_goal,
                scene_features.mean(dim=1),  # Average scene features
                subtask_probs[:, i, :].repeat(1, self.d_model // 2)  # Repeat subtask probs to match d_model
            ], dim=-1)

            # Generate low-level action
            action = self.low_level_generator(subtask_specific_features)
            low_level_actions.append(action)

            # Predict action parameters
            params = self.param_predictor(subtask_specific_features)
            action_params.append(params)

        # Stack actions and parameters
        actions = torch.stack(low_level_actions, dim=1)  # (batch_size, max_subtasks, action_space_dim)
        params = torch.stack(action_params, dim=1)       # (batch_size, max_subtasks, 20)

        return {
            'high_level_goal': interpreted_goal,
            'subtask_attention': subtask_probs,
            'low_level_actions': actions,
            'action_parameters': params,
            'subtask_sequence': self.generate_subtask_sequence(subtask_probs, actions)
        }

    def generate_subtask_sequence(self, subtask_probs, actions):
        """Generate sequence of subtasks based on attention probabilities"""
        # This would involve more sophisticated sequencing logic
        # For now, return a simple sequence based on highest probabilities
        subtask_sequence = []
        for i in range(subtask_probs.size(0)):  # batch dimension
            # Get subtasks with highest probabilities
            prob_sums = subtask_probs[i].sum(dim=1)  # Sum start/end probabilities
            sorted_indices = torch.argsort(prob_sums, descending=True)

            sequence = []
            for idx in sorted_indices:
                if prob_sums[idx] > 0.1:  # Threshold for inclusion
                    sequence.append({
                        'subtask_id': idx.item(),
                        'probability': prob_sums[idx].item(),
                        'action': actions[i, idx].tolist()
                    })

            subtask_sequence.append(sequence)

        return subtask_sequence

    def sample_action_sequence(self, action_probs, temperature: float = 1.0):
        """Sample action sequence from probability distributions"""
        # Apply temperature scaling
        scaled_probs = action_probs / temperature

        # Sample from the distribution
        sampled_actions = torch.multinomial(F.softmax(scaled_probs, dim=-1), 1)

        return sampled_actions
```

### Task and Motion Planning Integration

```python
class TaskMotionPlanner(nn.Module):
    def __init__(self, d_model: int = 768, action_space_dim: int = 6):
        super().__init__()
        self.d_model = d_model
        self.action_space_dim = action_space_dim

        # Task-level planner
        self.task_planner = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # Combined goal and scene features
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 50)  # 50 common high-level tasks
        )

        # Motion-level planner
        self.motion_planner = nn.Sequential(
            nn.Linear(d_model * 3, d_model),  # Goal, scene, and task features
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, action_space_dim * 10)  # 10 action steps
        )

        # Feasibility checker
        self.feasibility_checker = nn.Sequential(
            nn.Linear(d_model + action_space_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

        # Obstacle avoidance
        self.obstacle_avoider = ObstacleAvoidanceModule(d_model, action_space_dim)

    def forward(self, goal_features, scene_features, current_state):
        """
        Plan task and motion sequences
        Args:
            goal_features: High-level goal features
            scene_features: Scene understanding features
            current_state: Current robot state
        """
        batch_size = goal_features.size(0)

        # Plan high-level tasks
        task_features = torch.cat([goal_features, scene_features.mean(dim=1)], dim=-1)
        task_probs = F.softmax(self.task_planner(task_features), dim=-1)

        # For each possible task, plan motion sequence
        motion_sequences = []
        feasibility_scores = []

        for i in range(50):  # Iterate through possible tasks
            task_mask = F.one_hot(torch.tensor([i]), num_classes=50).float().to(task_probs.device)
            task_specific_features = torch.cat([
                goal_features,
                scene_features.mean(dim=1),
                task_mask.expand(batch_size, -1)
            ], dim=-1)

            # Plan motion sequence for this task
            motion_seq = self.motion_planner(task_specific_features)
            motion_seq = motion_seq.view(batch_size, 10, self.action_space_dim)

            # Check feasibility
            feasibility = self.check_feasibility(motion_seq, current_state)
            motion_sequences.append(motion_seq)
            feasibility_scores.append(feasibility)

        # Stack and select best sequence
        motion_sequences = torch.stack(motion_sequences, dim=1)  # (batch, 50, 10, action_dim)
        feasibility_scores = torch.stack(feasibility_scores, dim=1)  # (batch, 50)

        # Select sequence with highest feasibility
        best_task_idx = torch.argmax(feasibility_scores, dim=1)  # (batch,)
        best_motion_seq = torch.gather(
            motion_sequences,
            1,
            best_task_idx.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1, 1, 10, self.action_space_dim)
        ).squeeze(1)  # (batch, 10, action_dim)

        return {
            'task_probs': task_probs,
            'motion_sequence': best_motion_seq,
            'feasibility_scores': feasibility_scores,
            'selected_task': best_task_idx
        }

    def check_feasibility(self, motion_seq, current_state):
        """Check if motion sequence is feasible given current state"""
        batch_size, seq_len, action_dim = motion_seq.shape

        feasibility_scores = []
        for i in range(batch_size):
            seq_feasibility = []
            for j in range(seq_len):
                combined_features = torch.cat([
                    current_state[i].expand(1, -1),
                    motion_seq[i, j].unsqueeze(0)
                ], dim=-1)

                score = self.feasibility_checker(combined_features)
                seq_feasibility.append(score)

            # Average feasibility across sequence
            avg_feasibility = torch.mean(torch.stack(seq_feasibility))
            feasibility_scores.append(avg_feasibility)

        return torch.stack(feasibility_scores)

    def refine_motion_sequence(self, motion_seq, obstacles):
        """Refine motion sequence to avoid obstacles"""
        refined_seq = []

        for step in motion_seq:
            # Apply obstacle avoidance
            refined_action = self.obstacle_avoider.avoid_obstacles(step, obstacles)
            refined_seq.append(refined_action)

        return torch.stack(refined_seq, dim=0)

class ObstacleAvoidanceModule(nn.Module):
    def __init__(self, d_model: int, action_space_dim: int):
        super().__init__()
        self.d_model = d_model
        self.action_space_dim = action_space_dim

        # Obstacle detection and avoidance
        self.obstacle_detector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 20),  # Predict 20 obstacle parameters
            nn.Sigmoid()
        )

        # Avoidance action generator
        self.avoidance_generator = nn.Sequential(
            nn.Linear(action_space_dim + 20, d_model),  # Action + obstacle params
            nn.ReLU(),
            nn.Linear(d_model, action_space_dim)
        )

    def forward(self, current_action, obstacles):
        """Generate avoidance action based on obstacles"""
        # Process obstacles
        obstacle_features = self.obstacle_detector(obstacles)

        # Combine with current action
        combined = torch.cat([current_action, obstacle_features], dim=-1)

        # Generate avoidance-adjusted action
        avoidance_action = self.avoidance_generator(combined)

        return avoidance_action

    def avoid_obstacles(self, action, obstacles):
        """Apply obstacle avoidance to action"""
        # This would involve more sophisticated avoidance algorithms
        # For now, return the action with obstacle adjustments
        obstacle_positions = obstacles[:, :3]  # Assuming first 3 dims are position
        obstacle_radii = obstacles[:, 3]       # Assuming 4th dim is radius

        # Calculate distances to obstacles
        action_pos = action[:3]  # Assuming first 3 dims are position
        distances = torch.norm(obstacle_positions - action_pos.unsqueeze(0), dim=1)

        # Adjust action based on obstacle proximity
        adjustment = torch.zeros_like(action)
        for i, (dist, radius) in enumerate(zip(distances, obstacle_radii)):
            if dist < radius + 0.5:  # Safety margin
                # Calculate repulsive force
                direction = action_pos - obstacle_positions[i]
                magnitude = 1.0 / (dist + 1e-6)  # Repulsive force
                force = direction * magnitude * 0.1  # Scale factor

                adjustment[:3] += force  # Adjust position

        return action + adjustment
```

## Motor Control and Execution

### Continuous Control with Deep Reinforcement Learning

```python
class ContinuousControlActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, d_model: int = 768):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_model = d_model

        # State processing
        self.state_processor = nn.Sequential(
            nn.Linear(state_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU()
        )

        # Action generation
        self.action_generator = nn.Sequential(
            nn.Linear(d_model // 2 + d_model, d_model),  # Combined with language features
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, action_dim),
            nn.Tanh()  # Bound actions to [-1, 1]
        )

        # Action scaling
        self.action_scale = nn.Parameter(torch.ones(action_dim))

    def forward(self, state, language_features):
        """
        Generate continuous action from state and language features
        Args:
            state: Current robot state (batch_size, state_dim)
            language_features: Language features (batch_size, d_model)
        """
        # Process state
        state_features = self.state_processor(state)

        # Combine state and language features
        combined_features = torch.cat([state_features, language_features], dim=-1)

        # Generate action
        raw_action = self.action_generator(combined_features)

        # Scale action
        scaled_action = raw_action * self.action_scale

        return scaled_action

class ContinuousControlCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, d_model: int = 768):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_model = d_model

        # State-action value estimator
        self.value_estimator = nn.Sequential(
            nn.Linear(state_dim + action_dim + d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, state, action, language_features):
        """
        Estimate state-action value
        Args:
            state: Current robot state (batch_size, state_dim)
            action: Action to evaluate (batch_size, action_dim)
            language_features: Language features (batch_size, d_model)
        """
        # Combine state, action, and language features
        combined_features = torch.cat([state, action, language_features], dim=-1)

        # Estimate value
        value = self.value_estimator(combined_features)

        return value

class DDPGAgent(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, d_model: int = 768):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_model = d_model

        # Actor networks
        self.actor = ContinuousControlActor(state_dim, action_dim, d_model)
        self.actor_target = ContinuousControlActor(state_dim, action_dim, d_model)

        # Critic networks
        self.critic = ContinuousControlCritic(state_dim, action_dim, d_model)
        self.critic_target = ContinuousControlCritic(state_dim, action_dim, d_model)

        # Copy weights to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Noise for exploration
        self.noise = OrnsteinUhlenbeckNoise(action_dim)

    def select_action(self, state, language_features, add_noise=True):
        """Select action with optional exploration noise"""
        action = self.actor(state, language_features)

        if add_noise:
            noise = self.noise.sample()
            action = action + noise

        return torch.clamp(action, -1.0, 1.0)

    def update(self, replay_buffer, batch_size: int = 64, gamma: float = 0.99, tau: float = 0.005):
        """Update actor and critic networks"""
        # Sample batch from replay buffer
        state, action, reward, next_state, done, language_features = replay_buffer.sample(batch_size)

        # Compute target Q-value
        with torch.no_grad():
            next_action = self.actor_target(next_state, language_features)
            target_Q = self.critic_target(next_state, next_action, language_features)
            target_Q = reward + (1 - done) * gamma * target_Q

        # Critic loss
        current_Q = self.critic(state, action, language_features)
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Actor loss
        actor_loss = -self.critic(state, self.actor(state, language_features), language_features).mean()

        # Update networks
        # (In practice, these would be updated by an optimizer)

        # Update target networks
        self._soft_update(self.actor_target, self.actor, tau)
        self._soft_update(self.critic_target, self.critic, tau)

        return actor_loss.item(), critic_loss.item()

    def _soft_update(self, target_net, source_net, tau):
        """Soft update target network parameters"""
        for target_param, param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

class OrnsteinUhlenbeckNoise:
    def __init__(self, action_dim: int, mu: float = 0.0, theta: float = 0.15, sigma: float = 0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(action_dim) * self.mu

    def sample(self):
        """Sample noise from Ornstein-Uhlenbeck process"""
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state = self.state + dx
        return torch.FloatTensor(self.state)
```

### Model Predictive Control Integration

```python
class ModelPredictiveController(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, horizon: int = 10, d_model: int = 768):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.d_model = d_model

        # Dynamics model (learned system model)
        self.dynamics_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, state_dim)
        )

        # Cost function predictor
        self.cost_predictor = nn.Sequential(
            nn.Linear(state_dim + d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

        # Action sequence optimizer
        self.action_optimizer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, horizon * action_dim)
        )

    def forward(self, current_state, goal_features, scene_features):
        """
        Generate optimal action sequence using MPC
        Args:
            current_state: Current robot state (batch_size, state_dim)
            goal_features: Goal features (batch_size, d_model)
            scene_features: Scene features (batch_size, num_objects, d_model)
        """
        batch_size = current_state.size(0)

        # Predict optimal action sequence
        action_sequence = self.action_optimizer(goal_features).view(batch_size, self.horizon, self.action_dim)

        # Simulate trajectory using dynamics model
        simulated_states = []
        current_sim_state = current_state

        for t in range(self.horizon):
            # Predict next state
            next_state_input = torch.cat([current_sim_state, action_sequence[:, t, :]], dim=-1)
            next_state = self.dynamics_model(next_state_input)
            simulated_states.append(next_state)
            current_sim_state = next_state

        # Calculate costs for trajectory
        total_cost = 0
        for t in range(self.horizon):
            # Combine state and goal features for cost calculation
            state_goal_features = torch.cat([simulated_states[t], goal_features], dim=-1)
            step_cost = self.cost_predictor(state_goal_features)
            total_cost += step_cost

        return {
            'action_sequence': action_sequence,
            'predicted_trajectory': torch.stack(simulated_states, dim=1),
            'total_predicted_cost': total_cost,
            'first_action': action_sequence[:, 0, :]  # Return first action to execute
        }

    def optimize_trajectory(self, current_state, goal_features, scene_features, num_iterations: int = 10):
        """Iteratively optimize the action sequence"""
        # Initialize action sequence randomly
        action_sequence = torch.randn(current_state.size(0), self.horizon, self.action_dim).to(current_state.device)
        action_sequence = torch.clamp(action_sequence, -1, 1)

        for iteration in range(num_iterations):
            # Set requires_grad for optimization
            action_sequence.requires_grad_(True)

            # Forward pass to compute cost
            total_cost = self._compute_trajectory_cost(action_sequence, current_state, goal_features)

            # Backward pass to get gradients
            total_cost.backward()

            # Update action sequence using gradients
            with torch.no_grad():
                action_sequence = action_sequence - 0.01 * action_sequence.grad  # Learning rate
                action_sequence = torch.clamp(action_sequence, -1, 1)
                action_sequence.grad.zero_()

        return action_sequence

    def _compute_trajectory_cost(self, action_sequence, current_state, goal_features):
        """Compute total cost for a trajectory"""
        total_cost = 0
        current_sim_state = current_state

        for t in range(self.horizon):
            # Predict next state
            next_state_input = torch.cat([current_sim_state, action_sequence[:, t, :]], dim=-1)
            next_state = self.dynamics_model(next_state_input)

            # Calculate cost
            state_goal_features = torch.cat([next_state, goal_features], dim=-1)
            step_cost = self.cost_predictor(state_goal_features)
            total_cost += step_cost.mean()  # Average across batch

            current_sim_state = next_state

        return total_cost
```

## Imitation Learning for Action Generation

### Behavioral Cloning with Vision-Language Integration

```python
class ImitationLearningModule(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, d_model: int = 768):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_model = d_model

        # State processing
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU()
        )

        # Language instruction processing
        self.lang_encoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 2),
            nn.ReLU()
        )

        # Vision processing
        self.vision_encoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 2),
            nn.ReLU()
        )

        # Action generation from combined features
        self.action_generator = nn.Sequential(
            nn.Linear(d_model // 2 * 3, d_model),  # State + Lang + Vision
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, action_dim),
            nn.Tanh()
        )

        # Attention mechanism for instruction following
        self.instruction_attention = MultiHeadAttention(d_model // 2, 8)

    def forward(self, state, language_features, vision_features):
        """
        Generate action by imitating demonstrated behavior
        Args:
            state: Current robot state (batch_size, state_dim)
            language_features: Language instruction features (batch_size, seq_len, d_model)
            vision_features: Vision features (batch_size, num_objects, d_model)
        """
        batch_size = state.size(0)

        # Encode different modalities
        state_encoded = self.state_encoder(state).unsqueeze(1)  # (batch, 1, d_model//2)

        # Process language features with attention
        lang_encoded = self.lang_encoder(language_features.mean(dim=1)).unsqueeze(1)  # (batch, 1, d_model//2)

        # Process vision features
        vision_encoded = self.vision_encoder(vision_features.mean(dim=1)).unsqueeze(1)  # (batch, 1, d_model//2)

        # Combine all features
        combined_features = torch.cat([state_encoded, lang_encoded, vision_encoded], dim=-1)

        # Generate action
        action = self.action_generator(combined_features.squeeze(1))

        return action

    def compute_behavioral_cloning_loss(self, predicted_actions, expert_actions):
        """Compute loss for behavioral cloning"""
        return F.mse_loss(predicted_actions, expert_actions)

class MultiModalAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Projections for different modalities
        self.state_proj = nn.Linear(d_model, d_model)
        self.lang_proj = nn.Linear(d_model, d_model)
        self.vision_proj = nn.Linear(d_model, d_model)

        # Attention computation
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, state_features, lang_features, vision_features):
        """
        Multi-modal attention mechanism
        Args:
            state_features: State features (batch_size, seq_len, d_model)
            lang_features: Language features (batch_size, seq_len, d_model)
            vision_features: Vision features (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = state_features.shape

        # Project different modalities
        state_proj = self.state_proj(state_features)
        lang_proj = self.lang_proj(lang_features)
        vision_proj = self.vision_proj(vision_features)

        # Combine modalities
        combined_features = state_proj + lang_proj + vision_proj

        # Compute attention
        Q = self.q_proj(combined_features).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(combined_features).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(combined_features).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply attention
        attended_features = torch.matmul(attention_weights, V)
        attended_features = attended_features.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Output projection
        output = self.out_proj(attended_features)

        return output, attention_weights

class GAILDiscriminator(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, d_model: int = 768):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_model = d_model

        # Process state-action pairs
        self.state_action_processor = nn.Sequential(
            nn.Linear(state_dim + action_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU()
        )

        # Process language context
        self.lang_processor = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model // 8),
            nn.ReLU()
        )

        # Discriminator head
        self.discriminator_head = nn.Sequential(
            nn.Linear(d_model // 4 + d_model // 8, d_model // 8),
            nn.ReLU(),
            nn.Linear(d_model // 8, 1),
            nn.Sigmoid()
        )

    def forward(self, state, action, language_features):
        """
        Discriminate between expert and generated actions
        Args:
            state: Robot state (batch_size, state_dim)
            action: Action taken (batch_size, action_dim)
            language_features: Language context (batch_size, d_model)
        """
        # Process state-action
        state_action = torch.cat([state, action], dim=-1)
        sa_features = self.state_action_processor(state_action)

        # Process language
        lang_features = self.lang_processor(language_features)

        # Combine and classify
        combined = torch.cat([sa_features, lang_features], dim=-1)
        discrimination_score = self.discriminator_head(combined)

        return discrimination_score
```

## Advanced Control Strategies

### Adaptive Control with Uncertainty Quantification

```python
class UncertaintyAwareController(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, d_model: int = 768):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_model = d_model

        # Mean action predictor
        self.mean_predictor = nn.Sequential(
            nn.Linear(state_dim + d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, action_dim)
        )

        # Uncertainty predictor
        self.uncertainty_predictor = nn.Sequential(
            nn.Linear(state_dim + d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, action_dim),
            nn.Softplus()  # Ensure positive uncertainty
        )

        # Confidence-based action adjustment
        self.confidence_adjuster = nn.Sequential(
            nn.Linear(action_dim * 2, d_model),  # Mean + uncertainty
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, action_dim)
        )

    def forward(self, state, language_features):
        """
        Generate action with uncertainty quantification
        Args:
            state: Current robot state (batch_size, state_dim)
            language_features: Language features (batch_size, d_model)
        """
        # Combine state and language
        combined_features = torch.cat([state, language_features], dim=-1)

        # Predict mean action
        mean_action = self.mean_predictor(combined_features)

        # Predict uncertainty
        uncertainty = self.uncertainty_predictor(combined_features)

        # Adjust action based on confidence
        confidence_adjusted = self.confidence_adjuster(
            torch.cat([mean_action, uncertainty], dim=-1)
        )

        return {
            'mean_action': mean_action,
            'uncertainty': uncertainty,
            'confidence_adjusted_action': confidence_adjusted,
            'action_std': torch.sqrt(uncertainty + 1e-8)  # Standard deviation
        }

    def sample_action_with_uncertainty(self, state, language_features, num_samples: int = 10):
        """Sample actions considering uncertainty"""
        outputs = self.forward(state, language_features)

        # Sample from Gaussian distribution with predicted mean and variance
        sampled_actions = []
        for _ in range(num_samples):
            noise = torch.randn_like(outputs['action_std'])
            sampled_action = outputs['mean_action'] + outputs['action_std'] * noise
            sampled_actions.append(torch.clamp(sampled_action, -1, 1))

        return torch.stack(sampled_actions, dim=1)  # (batch, num_samples, action_dim)

class RobustController(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, d_model: int = 768):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_model = d_model

        # Nominal controller
        self.nominal_controller = nn.Sequential(
            nn.Linear(state_dim + d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, action_dim)
        )

        # Disturbance estimator
        self.disturbance_estimator = nn.Sequential(
            nn.Linear(state_dim + action_dim + d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, action_dim)
        )

        # Robust compensation
        self.robust_compensator = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, action_dim)
        )

    def forward(self, state, language_features, disturbance_estimate=None):
        """
        Generate robust control action
        Args:
            state: Current robot state (batch_size, state_dim)
            language_features: Language features (batch_size, d_model)
            disturbance_estimate: Estimated disturbance (optional)
        """
        # Combine state and language
        combined_features = torch.cat([state, language_features], dim=-1)

        # Nominal control action
        nominal_action = self.nominal_controller(combined_features)

        # Estimate disturbance if not provided
        if disturbance_estimate is None:
            disturbance_input = torch.cat([state, nominal_action, language_features], dim=-1)
            disturbance_estimate = self.disturbance_estimator(disturbance_input)

        # Robust compensation
        robust_compensation = self.robust_compensator(language_features)

        # Combined action
        robust_action = nominal_action - disturbance_estimate + robust_compensation

        return {
            'nominal_action': nominal_action,
            'disturbance_estimate': disturbance_estimate,
            'robust_compensation': robust_compensation,
            'robust_action': robust_action
        }

    def adapt_to_disturbance(self, state, action, next_state, language_features):
        """Adapt controller parameters based on observed disturbances"""
        # Predict next state with current model
        predicted_next_state = self.predict_next_state(state, action, language_features)

        # Calculate disturbance
        disturbance = next_state - predicted_next_state

        # Update disturbance estimator
        disturbance_input = torch.cat([state, action, language_features], dim=-1)
        current_disturbance_pred = self.disturbance_estimator(disturbance_input)

        # Compute adaptation loss
        adaptation_loss = F.mse_loss(current_disturbance_pred, disturbance)

        return adaptation_loss

    def predict_next_state(self, state, action, language_features):
        """Predict next state given current state, action, and context"""
        # This would use a learned dynamics model
        # For now, return a simplified prediction
        combined = torch.cat([state, action, language_features], dim=-1)
        next_state_delta = torch.tanh(torch.nn.Linear(combined.size(-1), self.state_dim)(combined))
        return state + next_state_delta
```

## Execution Monitoring and Adaptation

### Real-time Execution Monitoring

```python
class ExecutionMonitor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, d_model: int = 768):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_model = d_model

        # Execution state predictor
        self.execution_predictor = nn.Sequential(
            nn.Linear(state_dim + action_dim + d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, state_dim)
        )

        # Deviation detector
        self.deviation_detector = nn.Sequential(
            nn.Linear(state_dim * 2 + d_model, d_model),  # Expected + actual + context
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

        # Recovery action generator
        self.recovery_generator = nn.Sequential(
            nn.Linear(state_dim + d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, action_dim)
        )

        # Success predictor
        self.success_predictor = nn.Sequential(
            nn.Linear(state_dim + d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, current_state, expected_state, action, language_features):
        """
        Monitor execution and detect deviations
        Args:
            current_state: Actual current state (batch_size, state_dim)
            expected_state: Expected state after action (batch_size, state_dim)
            action: Action that was executed (batch_size, action_dim)
            language_features: Language context (batch_size, d_model)
        """
        # Predict expected next state
        expected_input = torch.cat([current_state, action, language_features], dim=-1)
        predicted_next_state = self.execution_predictor(expected_input)

        # Detect deviation
        deviation_input = torch.cat([expected_state, current_state, language_features], dim=-1)
        deviation_score = self.deviation_detector(deviation_input)

        # Predict success
        success_input = torch.cat([current_state, language_features], dim=-1)
        success_probability = self.success_predictor(success_input)

        # Generate recovery action if needed
        recovery_action = torch.zeros_like(action)
        needs_recovery = deviation_score > 0.5  # Threshold for recovery

        if needs_recovery.any():
            recovery_input = torch.cat([current_state, language_features], dim=-1)
            recovery_action = self.recovery_generator(recovery_input)

        return {
            'deviation_score': deviation_score,
            'success_probability': success_probability,
            'needs_recovery': needs_recovery,
            'recovery_action': recovery_action,
            'predicted_next_state': predicted_next_state
        }

    def detect_execution_failure(self, deviation_scores, threshold: float = 0.7):
        """Detect if execution has failed based on deviation scores"""
        return deviation_scores > threshold

    def compute_recovery_plan(self, current_state, goal_state, language_features):
        """Compute recovery plan when execution deviates"""
        # This would involve replanning
        # For now, return a simple corrective action
        correction = goal_state - current_state
        recovery_action = torch.tanh(correction)

        return recovery_action

class AdaptiveController(nn.Module):
    def __init__(self, base_controller: nn.Module, d_model: int = 768):
        super().__init__()
        self.base_controller = base_controller
        self.d_model = d_model

        # Adaptation network
        self.adaptation_network = nn.Sequential(
            nn.Linear(d_model + 2, d_model),  # Language features + deviation + success
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

        # Controller parameter adapter
        self.param_adapter = nn.Linear(d_model, d_model)

    def forward(self, state, language_features, deviation_score, success_prob):
        """
        Adapt controller based on execution feedback
        Args:
            state: Current state
            language_features: Language context
            deviation_score: Recent deviation score
            success_prob: Success probability
        """
        # Get base action
        if hasattr(self.base_controller, 'forward'):
            base_action = self.base_controller(state, language_features)
        else:
            # Handle different controller types
            base_action = self.base_controller(state)

        # Compute adaptation factor
        adaptation_input = torch.cat([
            language_features.mean(dim=1),  # Average over sequence
            deviation_score,
            success_prob
        ], dim=-1)

        adaptation_factor = self.adaptation_network(adaptation_input)

        # Adapt controller parameters
        adapted_params = self.param_adapter(language_features.mean(dim=1))

        # Modify action based on adaptation
        if isinstance(base_action, dict):
            # If base controller returns a dictionary
            adapted_action = {
                'action': base_action['action'] * adaptation_factor.unsqueeze(1),
                'adaptation_factor': adaptation_factor
            }
        else:
            # If base controller returns a tensor
            adapted_action = base_action * adaptation_factor.unsqueeze(1)

        return adapted_action, adaptation_factor
```

## Quality Assessment for Action Generation

### Action Quality Metrics

```python
class ActionQualityAssessor:
    def __init__(self):
        self.metrics = {}

    def assess_action_feasibility(self, action, state, constraints):
        """Assess whether an action is physically feasible"""
        # Check joint limits
        joint_limits_violated = self.check_joint_limits(action, constraints.get('joint_limits', {}))

        # Check collision constraints
        collision_risk = self.assess_collision_risk(action, state, constraints.get('collision_objects', []))

        # Check dynamic constraints
        dynamic_feasible = self.check_dynamic_feasibility(action, constraints.get('dynamics', {}))

        feasibility_score = self.calculate_feasibility_score(
            joint_limits_violated, collision_risk, dynamic_feasible
        )

        return {
            'feasibility_score': feasibility_score,
            'joint_limits_ok': not joint_limits_violated,
            'collision_risk': collision_risk,
            'dynamically_feasible': dynamic_feasible
        }

    def check_joint_limits(self, action, joint_limits):
        """Check if action violates joint limits"""
        if not joint_limits:
            return False

        for joint_idx, (min_limit, max_limit) in joint_limits.items():
            if action[joint_idx] < min_limit or action[joint_idx] > max_limit:
                return True  # Violation detected

        return False  # No violations

    def assess_collision_risk(self, action, state, collision_objects):
        """Assess collision risk for the action"""
        if not collision_objects:
            return 0.0  # No collision objects, no risk

        # This would involve forward kinematics and collision checking
        # For now, return a mock assessment
        risk_score = 0.0
        for obj in collision_objects:
            # Calculate distance to collision object after action
            future_pos = state[:3] + action[:3]  # Simplified: assume first 3 dims are position
            dist_to_obj = np.linalg.norm(future_pos - obj['position'])
            if dist_to_obj < obj['radius']:
                risk_score += 1.0  # High risk if within object radius

        return min(risk_score / len(collision_objects), 1.0)  # Normalize

    def check_dynamic_feasibility(self, action, dynamics_constraints):
        """Check if action is dynamically feasible"""
        # Check velocity limits
        vel_limits = dynamics_constraints.get('velocity_limits', [float('inf')] * len(action))
        for i, (act, max_vel) in enumerate(zip(action, vel_limits)):
            if abs(act) > max_vel:
                return False

        # Check acceleration limits
        acc_limits = dynamics_constraints.get('acceleration_limits', [float('inf')] * len(action))
        # This would require comparing to previous action for acceleration

        return True

    def calculate_feasibility_score(self, joint_violation, collision_risk, dynamic_feasible):
        """Calculate overall feasibility score"""
        joint_penalty = 1.0 if joint_violation else 0.0
        collision_penalty = collision_risk
        dynamic_penalty = 0.0 if dynamic_feasible else 1.0

        # Weighted combination (weights can be tuned)
        penalty = 0.4 * joint_penalty + 0.4 * collision_penalty + 0.2 * dynamic_penalty
        return 1.0 - penalty  # Convert penalty to score

    def evaluate_task_completion(self, action_sequence, initial_state, goal_state, environment):
        """Evaluate how well an action sequence achieves the goal"""
        current_state = initial_state.copy()

        # Execute action sequence
        for action in action_sequence:
            current_state = environment.step(current_state, action)

        # Calculate distance to goal
        goal_distance = np.linalg.norm(current_state - goal_state)

        # Calculate trajectory efficiency
        trajectory_length = sum(np.linalg.norm(action) for action in action_sequence)

        # Calculate success metrics
        success_threshold = 0.1  # Distance threshold for success
        is_successful = goal_distance < success_threshold

        return {
            'final_distance_to_goal': goal_distance,
            'trajectory_length': trajectory_length,
            'efficiency_ratio': goal_distance / trajectory_length if trajectory_length > 0 else float('inf'),
            'is_successful': is_successful,
            'success_rate': 1.0 if is_successful else 0.0
        }

    def assess_instruction_following(self, generated_actions, target_instruction, scene_context):
        """Assess how well generated actions follow the instruction"""
        # This would involve complex natural language understanding
        # For now, return a mock assessment

        # Calculate action appropriateness based on instruction semantics
        instruction_semantics = self.extract_instruction_semantics(target_instruction)
        action_semantics = self.extract_action_semantics(generated_actions)

        semantic_alignment = self.calculate_semantic_alignment(instruction_semantics, action_semantics)

        # Calculate spatial appropriateness
        spatial_alignment = self.calculate_spatial_alignment(generated_actions, scene_context)

        # Combine metrics
        instruction_following_score = 0.6 * semantic_alignment + 0.4 * spatial_alignment

        return {
            'instruction_following_score': instruction_following_score,
            'semantic_alignment': semantic_alignment,
            'spatial_alignment': spatial_alignment,
            'action_appropriateness': instruction_following_score
        }

    def extract_instruction_semantics(self, instruction):
        """Extract semantic meaning from instruction"""
        # This would use NLP techniques to parse the instruction
        # For now, return a mock semantic representation
        return {'action': 'move', 'object': 'box', 'destination': 'table'}

    def extract_action_semantics(self, actions):
        """Extract semantic meaning from action sequence"""
        # This would analyze the action sequence for semantic content
        # For now, return a mock semantic representation
        return {'movement_type': 'translation', 'magnitude': 0.5, 'direction': 'positive_x'}

    def calculate_semantic_alignment(self, instruction_semantics, action_semantics):
        """Calculate alignment between instruction and action semantics"""
        # This would involve semantic similarity computation
        # For now, return a mock alignment score
        return 0.8  # High alignment

    def calculate_spatial_alignment(self, actions, scene_context):
        """Calculate spatial alignment between actions and scene context"""
        # This would involve checking if actions are spatially appropriate
        # For now, return a mock alignment score
        return 0.9  # High spatial alignment

    def evaluate_execution_robustness(self, action_sequence, disturbance_levels):
        """Evaluate how robust the action sequence is to disturbances"""
        success_rates = []

        for disturbance_level in disturbance_levels:
            disturbed_success_rate = self.test_disturbed_execution(
                action_sequence, disturbance_level
            )
            success_rates.append(disturbed_success_rate)

        average_robustness = sum(success_rates) / len(success_rates) if success_rates else 0.0

        return {
            'robustness_scores': success_rates,
            'average_robustness': average_robustness,
            'disturbance_tolerance': self.calculate_disturbance_tolerance(success_rates, disturbance_levels)
        }

    def test_disturbed_execution(self, action_sequence, disturbance_level):
        """Test execution under disturbance"""
        # This would simulate execution with added disturbances
        # For now, return a mock success rate
        return max(0.0, 1.0 - disturbance_level)  # Simple degradation model

    def calculate_disturbance_tolerance(self, success_rates, disturbance_levels):
        """Calculate disturbance tolerance metric"""
        if not success_rates or not disturbance_levels:
            return 0.0

        # Find the disturbance level at which success rate drops below 50%
        for i, (success_rate, dist_level) in enumerate(zip(success_rates, disturbance_levels)):
            if success_rate < 0.5:
                return dist_level

        # If success rate stays above 50%, return the highest tested level
        return max(disturbance_levels) if disturbance_levels else 0.0

    def comprehensive_action_assessment(self, action_sequence, environment, instruction, scene_context):
        """Comprehensive assessment of action quality"""
        assessment = {}

        # Feasibility assessment
        if len(action_sequence) > 0:
            first_action = action_sequence[0]
            assessment['feasibility'] = self.assess_action_feasibility(
                first_action, environment.get_current_state(), environment.get_constraints()
            )

        # Task completion assessment
        assessment['task_completion'] = self.evaluate_task_completion(
            action_sequence, environment.get_initial_state(),
            environment.get_goal_state(), environment
        )

        # Instruction following assessment
        assessment['instruction_following'] = self.assess_instruction_following(
            action_sequence, instruction, scene_context
        )

        # Robustness assessment
        assessment['robustness'] = self.evaluate_execution_robustness(
            action_sequence, [0.1, 0.2, 0.3]
        )

        # Overall quality score
        assessment['overall_quality'] = self.calculate_overall_quality(assessment)

        return assessment

    def calculate_overall_quality(self, assessment_results):
        """Calculate overall action quality score"""
        scores = []

        if 'feasibility' in assessment_results:
            scores.append(assessment_results['feasibility']['feasibility_score'])

        if 'task_completion' in assessment_results:
            scores.append(assessment_results['task_completion']['success_rate'])

        if 'instruction_following' in assessment_results:
            scores.append(assessment_results['instruction_following']['instruction_following_score'])

        if 'robustness' in assessment_results:
            scores.append(assessment_results['robustness']['average_robustness'])

        return sum(scores) / len(scores) if scores else 0.0
```

## Key Takeaways

- Action generation integrates perception, language understanding, and motor control
- Hierarchical planning decomposes complex goals into manageable subtasks
- Continuous control systems enable precise motor execution
- Imitation learning allows robots to learn from expert demonstrations
- Uncertainty quantification improves robustness in uncertain environments
- Execution monitoring detects and corrects deviations
- Quality assessment ensures reliable action execution
- Real-time adaptation enables response to changing conditions

## Next Steps

In the next chapter, we'll explore multimodal integration and system architecture, learning how to combine all the components we've studied into a cohesive Vision-Language-Action system that can operate effectively in real-world environments.