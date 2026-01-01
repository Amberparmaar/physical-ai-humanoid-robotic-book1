---
sidebar_position: 3
title: Planning and Decision Making
---

# Planning and Decision Making

In this chapter, we'll explore the critical components of planning and decision-making in AI robot brains. Planning involves determining sequences of actions to achieve goals, while decision-making involves selecting the best course of action from available options based on current state and objectives.

## Understanding Planning in Robotics

Planning in robotics can be categorized into several types:

### 1. Motion Planning
- Path planning to navigate from start to goal
- Trajectory generation for smooth motion
- Obstacle avoidance and collision detection
- Dynamic replanning when environment changes

### 2. Task Planning
- High-level task decomposition
- Sequence planning for complex behaviors
- Resource allocation and scheduling
- Multi-objective optimization

### 3. Contingency Planning
- Handling unexpected situations
- Backup plans and fallback behaviors
- Risk assessment and mitigation
- Adaptive planning based on feedback

## Classical Planning Approaches

### A* Path Planning

```python
import heapq
import numpy as np
from typing import List, Tuple, Optional

class AStarPlanner:
    def __init__(self, grid: np.ndarray):
        self.grid = grid
        self.height, self.width = grid.shape

    def plan_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Plan path using A* algorithm"""
        # Priority queue: (f_score, g_score, position)
        open_set = [(0, 0, start)]
        closed_set = set()

        # Cost dictionaries
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        # Path reconstruction
        came_from = {}

        while open_set:
            current_f, current_g, current = heapq.heappop(open_set)

            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            if current in closed_set:
                continue

            closed_set.add(current)

            # Check neighbors
            for neighbor in self.get_neighbors(current):
                if neighbor in closed_set:
                    continue

                tentative_g = g_score[current] + self.distance(current, neighbor)

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], g_score[neighbor], neighbor))

        return []  # No path found

    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Calculate heuristic distance (Manhattan distance)"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Calculate distance between two points"""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighbors for path planning"""
        neighbors = []
        x, y = pos

        # 8-directional movement
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue  # Skip current position

                nx, ny = x + dx, y + dy

                # Check bounds and obstacles
                if (0 <= nx < self.width and
                    0 <= ny < self.height and
                    self.grid[ny, nx] == 0):  # 0 = free space, 1 = obstacle
                    neighbors.append((nx, ny))

        return neighbors

    def smooth_path(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Smooth path by removing unnecessary waypoints"""
        if len(path) < 3:
            return path

        smoothed_path = [path[0]]
        i = 0

        while i < len(path) - 1:
            j = i + 1
            # Find the farthest point that can be reached without obstacle
            while j < len(path) - 1 and self.is_line_clear(path[i], path[j]):
                j += 1

            smoothed_path.append(path[j-1])
            i = j - 1

        if smoothed_path[-1] != path[-1]:
            smoothed_path.append(path[-1])

        return smoothed_path

    def is_line_clear(self, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        """Check if line between two points is clear of obstacles"""
        x0, y0 = start
        x1, y1 = end

        # Bresenham's line algorithm
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0

        while True:
            if self.grid[y, x] == 1:  # Obstacle found
                return False

            if x == x1 and y == y1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

        return True
```

### Dijkstra's Algorithm for Weighted Graphs

```python
import heapq
from typing import Dict, List, Tuple

class DijkstraPlanner:
    def __init__(self):
        self.graph = {}  # Adjacency list representation

    def add_edge(self, from_node: str, to_node: str, weight: float):
        """Add weighted edge to graph"""
        if from_node not in self.graph:
            self.graph[from_node] = []
        if to_node not in self.graph:
            self.graph[to_node] = []

        self.graph[from_node].append((to_node, weight))
        self.graph[to_node].append((from_node, weight))  # Undirected graph

    def find_shortest_path(self, start: str, goal: str) -> Tuple[List[str], float]:
        """Find shortest path using Dijkstra's algorithm"""
        # Priority queue: (distance, node)
        pq = [(0, start)]
        distances = {start: 0}
        previous = {start: None}
        visited = set()

        while pq:
            current_dist, current = heapq.heappop(pq)

            if current in visited:
                continue

            visited.add(current)

            if current == goal:
                # Reconstruct path
                path = []
                node = goal
                while node is not None:
                    path.append(node)
                    node = previous[node]
                path.reverse()
                return path, current_dist

            for neighbor, weight in self.graph.get(current, []):
                if neighbor not in visited:
                    new_dist = current_dist + weight
                    if neighbor not in distances or new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        previous[neighbor] = current
                        heapq.heappush(pq, (new_dist, neighbor))

        return [], float('inf')  # No path found
```

## Sampling-Based Motion Planning

### RRT (Rapidly-exploring Random Tree)

```python
import numpy as np
from typing import List, Tuple, Optional
import random

class RRTPlanner:
    def __init__(self, bounds: Tuple[float, float, float, float], step_size: float = 1.0):
        self.bounds = bounds  # (x_min, x_max, y_min, y_max)
        self.step_size = step_size
        self.tree = []  # List of nodes in the tree

    def plan_path(self, start: np.ndarray, goal: np.ndarray,
                  obstacles: List[np.ndarray], max_iterations: int = 10000) -> List[np.ndarray]:
        """Plan path using RRT algorithm"""
        self.tree = [start]
        goal_found = False

        for iteration in range(max_iterations):
            # Sample random point
            if random.random() < 0.05:  # 5% chance to sample goal
                rand_point = goal
            else:
                rand_point = self.sample_free_space()

            # Find nearest node in tree
            nearest_idx = self.nearest_node(rand_point)
            nearest_node = self.tree[nearest_idx]

            # Extend towards random point
            new_node = self.steer(nearest_node, rand_point)

            # Check collision
            if self.is_collision_free(new_node, obstacles):
                self.tree.append(new_node)

                # Check if goal is reached
                if np.linalg.norm(new_node - goal) < self.step_size:
                    goal_found = True
                    break

        if goal_found:
            return self.extract_path(goal)
        else:
            return []  # No path found

    def sample_free_space(self) -> np.ndarray:
        """Sample a random point in free space"""
        x = random.uniform(self.bounds[0], self.bounds[1])
        y = random.uniform(self.bounds[2], self.bounds[3])
        return np.array([x, y])

    def nearest_node(self, point: np.ndarray) -> int:
        """Find index of nearest node in tree"""
        distances = [np.linalg.norm(point - node) for node in self.tree]
        return np.argmin(distances)

    def steer(self, from_node: np.ndarray, to_point: np.ndarray) -> np.ndarray:
        """Steer from from_node towards to_point"""
        direction = to_point - from_node
        distance = np.linalg.norm(direction)

        if distance <= self.step_size:
            return to_point
        else:
            normalized_dir = direction / distance
            return from_node + normalized_dir * self.step_size

    def is_collision_free(self, point: np.ndarray, obstacles: List[np.ndarray]) -> bool:
        """Check if point is collision-free"""
        # Check bounds
        if (point[0] < self.bounds[0] or point[0] > self.bounds[1] or
            point[1] < self.bounds[2] or point[1] > self.bounds[3]):
            return False

        # Check obstacles (simplified - assuming circular obstacles)
        for obstacle in obstacles:
            if np.linalg.norm(point - obstacle[:2]) < obstacle[2]:  # obstacle[2] = radius
                return False

        return True

    def extract_path(self, goal: np.ndarray) -> List[np.ndarray]:
        """Extract path from tree to goal"""
        # In a complete RRT implementation, we would store parent-child relationships
        # For this example, we'll return a simple path
        path = [goal]
        current = goal

        # Find the path by looking for nodes in the tree that lead to the goal
        # This is simplified - in practice, you'd store the tree structure
        for node in reversed(self.tree):
            if np.linalg.norm(current - node) < self.step_size * 1.5:
                path.append(node)
                current = node

        path.reverse()
        return path
```

## Decision Making Under Uncertainty

### Markov Decision Process (MDP)

```python
import numpy as np
from typing import Dict, List, Tuple

class MDPPlanner:
    def __init__(self, states: List[str], actions: List[str],
                 transition_probs: Dict, rewards: Dict, gamma: float = 0.9):
        self.states = states
        self.actions = actions
        self.transition_probs = transition_probs  # P(s'|s,a)
        self.rewards = rewards  # R(s,a,s')
        self.gamma = gamma  # Discount factor

    def value_iteration(self, max_iterations: int = 1000, threshold: float = 1e-6) -> Dict[str, float]:
        """Solve MDP using value iteration"""
        # Initialize value function
        V = {state: 0.0 for state in self.states}

        for iteration in range(max_iterations):
            V_new = V.copy()
            delta = 0.0

            for state in self.states:
                # Calculate value for each action
                action_values = []
                for action in self.actions:
                    value = 0.0
                    for next_state in self.states:
                        prob = self.transition_probs.get((state, action, next_state), 0.0)
                        reward = self.rewards.get((state, action, next_state), 0.0)
                        value += prob * (reward + self.gamma * V[next_state])
                    action_values.append(value)

                # Take maximum value
                V_new[state] = max(action_values) if action_values else 0.0
                delta = max(delta, abs(V_new[state] - V[state]))

            V = V_new

            if delta < threshold:
                break

        return V

    def extract_policy(self, V: Dict[str, float]) -> Dict[str, str]:
        """Extract optimal policy from value function"""
        policy = {}

        for state in self.states:
            action_values = []
            for action in self.actions:
                value = 0.0
                for next_state in self.states:
                    prob = self.transition_probs.get((state, action, next_state), 0.0)
                    reward = self.rewards.get((state, action, next_state), 0.0)
                    value += prob * (reward + self.gamma * V[next_state])
                action_values.append((action, value))

            # Choose action with maximum value
            best_action = max(action_values, key=lambda x: x[1])[0]
            policy[state] = best_action

        return policy
```

### Partially Observable MDP (POMDP)

```python
class POMDPPlanner:
    def __init__(self, states: List[str], actions: List[str], observations: List[str],
                 transition_probs: Dict, observation_probs: Dict, rewards: Dict, gamma: float = 0.9):
        self.states = states
        self.actions = actions
        self.observations = observations
        self.transition_probs = transition_probs  # P(s'|s,a)
        self.observation_probs = observation_probs  # P(o|s',a)
        self.rewards = rewards  # R(s,a,s')
        self.gamma = gamma

    def belief_update(self, belief: Dict[str, float], action: str, observation: str) -> Dict[str, float]:
        """Update belief state based on action and observation"""
        new_belief = {}

        # Predict step: P(s'|b,a) = sum_s P(s'|s,a) * b(s)
        predicted_belief = {}
        for next_state in self.states:
            prob = 0.0
            for current_state in self.states:
                prob += (self.transition_probs.get((current_state, action, next_state), 0.0) *
                        belief.get(current_state, 0.0))
            predicted_belief[next_state] = prob

        # Update step: b'(s') = P(o|s',a) * P(s'|b,a) / P(o|b,a)
        normalization = 0.0
        for state in self.states:
            obs_prob = self.observation_probs.get((state, action, observation), 0.0)
            new_belief[state] = obs_prob * predicted_belief[state]
            normalization += new_belief[state]

        # Normalize
        if normalization > 0:
            for state in self.states:
                new_belief[state] /= normalization

        return new_belief

    def expected_utility(self, belief: Dict[str, float], action: str) -> float:
        """Calculate expected utility of taking an action in a belief state"""
        utility = 0.0

        # Sum over all possible next states
        for next_state in self.states:
            state_prob = 0.0
            for current_state in self.states:
                state_prob += (self.transition_probs.get((current_state, action, next_state), 0.0) *
                             belief.get(current_state, 0.0))

            # Expected reward for this transition
            expected_reward = 0.0
            for outcome_state in self.states:
                expected_reward += (self.rewards.get((next_state, action, outcome_state), 0.0) *
                                  self.transition_probs.get((next_state, action, outcome_state), 0.0))

            utility += state_prob * expected_reward

        return utility
```

## Task Planning and Hierarchical Decision Making

### Hierarchical Task Network (HTN) Planner

```python
from typing import List, Dict, Any, Callable
from dataclasses import dataclass

@dataclass
class Task:
    name: str
    args: List[Any]
    is_primitive: bool = False

@dataclass
class Method:
    name: str
    task: str
    preconditions: List[str]
    subtasks: List[Task]

class HTNPlanner:
    def __init__(self):
        self.primitive_tasks = {}  # name -> function
        self.methods = {}  # task_name -> list of methods

    def add_primitive_task(self, name: str, func: Callable):
        """Add a primitive task (leaf in task hierarchy)"""
        self.primitive_tasks[name] = func

    def add_method(self, method: Method):
        """Add a method for decomposing a compound task"""
        if method.task not in self.methods:
            self.methods[method.task] = []
        self.methods[method.task].append(method)

    def plan(self, task: Task, state: Dict[str, Any]) -> List[Task]:
        """Generate plan for the given task"""
        return self._decompose_task(task, state)

    def _decompose_task(self, task: Task, state: Dict[str, Any]) -> List[Task]:
        """Decompose a task into subtasks or execute primitive task"""
        if task.is_primitive:
            return [task]

        # Find applicable methods
        applicable_methods = self._find_applicable_methods(task, state)

        for method in applicable_methods:
            # Check preconditions
            if self._check_preconditions(method.preconditions, state):
                # Decompose using this method
                plan = []
                for subtask in method.subtasks:
                    subplan = self._decompose_task(subtask, state)
                    plan.extend(subplan)
                return plan

        # No applicable method found
        return []

    def _find_applicable_methods(self, task: Task, state: Dict[str, Any]) -> List[Method]:
        """Find methods that can decompose the given task"""
        if task.name in self.methods:
            return self.methods[task.name]
        return []

    def _check_preconditions(self, preconditions: List[str], state: Dict[str, Any]) -> bool:
        """Check if preconditions are satisfied in the current state"""
        for condition in preconditions:
            # Simple string-based condition checking
            # In practice, this would be more sophisticated
            if condition not in state.get('facts', []):
                return False
        return True
```

## Reinforcement Learning for Planning

### Q-Learning Implementation

```python
import numpy as np
from collections import defaultdict

class QLearningPlanner:
    def __init__(self, actions: List[int], learning_rate: float = 0.1,
                 discount_factor: float = 0.95, exploration_rate: float = 0.1):
        self.actions = actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table = defaultdict(lambda: np.zeros(len(actions)))

    def get_action(self, state: str) -> int:
        """Choose action using epsilon-greedy policy"""
        if np.random.random() < self.exploration_rate:
            # Explore: random action
            return np.random.choice(self.actions)
        else:
            # Exploit: best action according to Q-table
            q_values = self.q_table[state]
            return self.actions[np.argmax(q_values)]

    def update_q_value(self, state: str, action: int, reward: float, next_state: str):
        """Update Q-value using Q-learning update rule"""
        current_q = self.q_table[state][self.actions.index(action)]

        # Get maximum Q-value for next state
        next_max_q = np.max(self.q_table[next_state])

        # Q-learning update
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * next_max_q - current_q
        )

        self.q_table[state][self.actions.index(action)] = new_q

    def learn_episode(self, env, max_steps: int = 1000):
        """Learn from a single episode"""
        state = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action = self.get_action(str(state))
            next_state, reward, done, info = env.step(action)

            self.update_q_value(str(state), action, reward, str(next_state))

            state = next_state
            total_reward += reward

            if done:
                break

        return total_reward
```

### Deep Q-Network (DQN) for Continuous Environments

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Neural networks
        self.q_network = DQN(state_size, action_size).to(self.device)
        self.target_network = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Update target network
        self.update_target_network()

    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())

    def replay(self, batch_size: int = 32):
        """Train the model on a batch of experiences"""
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.99 * next_q_values * ~dones)

        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
```

## Multi-Agent Planning and Coordination

### Decentralized Multi-Agent Planning

```python
class MultiAgentPlanner:
    def __init__(self, num_agents: int):
        self.num_agents = num_agents
        self.agent_plans = [None] * num_agents
        self.conflicts = []

    def coordinate_agents(self, agent_goals: List[Tuple[int, int]],
                         environment_grid: np.ndarray) -> List[List[Tuple[int, int]]]:
        """Coordinate multiple agents to avoid conflicts"""
        # Plan for each agent independently first
        individual_plans = []
        for i in range(self.num_agents):
            planner = AStarPlanner(environment_grid)
            start = self.get_agent_position(i)  # Assume we know starting positions
            plan = planner.plan_path(start, agent_goals[i])
            individual_plans.append(plan)

        # Detect and resolve conflicts
        coordinated_plans = self.resolve_conflicts(individual_plans)

        return coordinated_plans

    def get_agent_position(self, agent_id: int) -> Tuple[int, int]:
        """Get current position of an agent (simplified)"""
        # In a real implementation, this would query the actual agent positions
        return (agent_id, agent_id)

    def resolve_conflicts(self, plans: List[List[Tuple[int, int]]]) -> List[List[Tuple[int, int]]]:
        """Resolve conflicts between agent plans"""
        max_time_steps = max(len(plan) for plan in plans) if plans else 0

        # Create space-time occupancy grid
        occupancy = {}  # (x, y, t) -> agent_id

        for agent_id, plan in enumerate(plans):
            for t, pos in enumerate(plan):
                occupancy[(pos[0], pos[1], t)] = agent_id

        # Resolve conflicts by adjusting plans
        resolved_plans = [plan.copy() for plan in plans]

        # Simple conflict resolution: wait for other agents to pass
        for agent_id, plan in enumerate(resolved_plans):
            for t in range(len(plan)):
                pos = plan[t]
                conflict_time = self.find_conflict_time(pos, t, occupancy, agent_id)

                if conflict_time != -1:
                    # Insert wait time
                    wait_start = t
                    while (pos[0], pos[1], wait_start) in occupancy:
                        wait_start += 1

                    # Adjust plan to wait
                    for i in range(t, wait_start):
                        resolved_plans[agent_id].insert(t, plan[t])

        return resolved_plans

    def find_conflict_time(self, pos: Tuple[int, int], time: int,
                          occupancy: Dict, current_agent: int) -> int:
        """Find if there's a conflict at position and time"""
        if (pos[0], pos[1], time) in occupancy:
            if occupancy[(pos[0], pos[1], time)] != current_agent:
                return time
        return -1
```

## Real-Time Planning and Replanning

### Dynamic Window Approach (DWA) for Local Planning

```python
import numpy as np
from typing import List, Tuple

class DWAPlanner:
    def __init__(self, robot_radius: float = 0.5, max_speed: float = 1.0,
                 max_angular_speed: float = 1.0, dt: float = 0.1):
        self.robot_radius = robot_radius
        self.max_speed = max_speed
        self.max_angular_speed = max_angular_speed
        self.dt = dt
        self.prediction_time = 2.0  # seconds to predict ahead

    def plan_local_path(self, current_pose: Tuple[float, float, float],
                       current_vel: Tuple[float, float],
                       goal: Tuple[float, float],
                       obstacles: List[Tuple[float, float]]) -> Tuple[float, float]:
        """Plan local velocity commands using Dynamic Window Approach"""
        # Define velocity space
        v_min = max(-self.max_speed, current_vel[0] - 0.5 * self.dt)
        v_max = min(self.max_speed, current_vel[0] + 0.5 * self.dt)
        w_min = max(-self.max_angular_speed, current_vel[1] - 1.0 * self.dt)
        w_max = min(self.max_angular_speed, current_vel[1] + 1.0 * self.dt)

        # Sample velocities in the window
        v_samples = np.linspace(v_min, v_max, 20)
        w_samples = np.linspace(w_min, w_max, 20)

        best_score = float('-inf')
        best_vel = (0.0, 0.0)

        for v in v_samples:
            for w in w_samples:
                # Simulate trajectory
                traj = self.simulate_trajectory(current_pose, (v, w))

                # Evaluate trajectory
                score = self.evaluate_trajectory(traj, goal, obstacles)

                if score > best_score:
                    best_score = score
                    best_vel = (v, w)

        return best_vel

    def simulate_trajectory(self, start_pose: Tuple[float, float, float],
                           velocity: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Simulate robot trajectory for given velocity"""
        x, y, theta = start_pose
        v, w = velocity

        trajectory = []
        time_steps = int(self.prediction_time / self.dt)

        for i in range(time_steps):
            # Update pose using motion model
            if abs(w) < 1e-6:  # Straight line motion
                x += v * self.dt * np.cos(theta)
                y += v * self.dt * np.sin(theta)
            else:  # Circular motion
                x += (v / w) * (np.sin(theta + w * self.dt) - np.sin(theta))
                y += (v / w) * (np.cos(theta) - np.cos(theta + w * self.dt))
                theta += w * self.dt

            trajectory.append((x, y))

        return trajectory

    def evaluate_trajectory(self, trajectory: List[Tuple[float, float]],
                           goal: Tuple[float, float],
                           obstacles: List[Tuple[float, float]]) -> float:
        """Evaluate trajectory based on multiple criteria"""
        if not trajectory:
            return float('-inf')

        # Check for collisions
        for point in trajectory:
            for obs in obstacles:
                if np.linalg.norm(np.array(point) - np.array(obs)) < self.robot_radius:
                    return float('-inf')  # Collision

        # Calculate goal distance (should be minimized)
        final_pos = trajectory[-1]
        goal_dist = np.linalg.norm(np.array(final_pos) - np.array(goal))
        goal_score = -goal_dist  # Negative because we want to minimize distance

        # Calculate trajectory length (prefer shorter paths)
        path_length = 0.0
        for i in range(1, len(trajectory)):
            path_length += np.linalg.norm(
                np.array(trajectory[i]) - np.array(trajectory[i-1])
            )
        length_score = -path_length

        # Calculate clearance from obstacles
        min_clearance = float('inf')
        for point in trajectory:
            for obs in obstacles:
                dist = np.linalg.norm(np.array(point) - np.array(obs)) - self.robot_radius
                min_clearance = min(min_clearance, dist)
        clearance_score = min_clearance

        # Weighted combination of scores
        total_score = 0.6 * goal_score + 0.2 * length_score + 0.2 * clearance_score

        return total_score
```

## Decision Making with NVIDIA Isaac

### Isaac ROS Planning Components

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan
import numpy as np

class IsaacPlanningNode(Node):
    def __init__(self):
        super().__init__('isaac_planning_node')

        # Publishers and subscribers
        self.goal_sub = self.create_subscription(
            PoseStamped, '/move_base_simple/goal', self.goal_callback, 10)

        self.laser_sub = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10)

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.local_plan_pub = self.create_publisher(Path, '/local_plan', 10)
        self.global_plan_pub = self.create_publisher(Path, '/global_plan', 10)

        # Planning components
        self.global_planner = AStarPlanner(np.zeros((100, 100)))  # Simplified grid
        self.local_planner = DWAPlanner()

        self.current_goal = None
        self.obstacles = []

    def goal_callback(self, msg: PoseStamped):
        """Handle new goal from user"""
        goal_x = int(msg.pose.position.x)
        goal_y = int(msg.pose.position.y)

        # Update global plan
        if self.current_goal != (goal_x, goal_y):
            self.current_goal = (goal_x, goal_y)
            self.plan_global_path()

    def laser_callback(self, msg: LaserScan):
        """Process laser scan data for obstacle detection"""
        # Convert laser scan to obstacle positions
        self.obstacles = []
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment

        for i, range_val in enumerate(msg.ranges):
            if not np.isnan(range_val) and range_val < msg.range_max:
                angle = angle_min + i * angle_increment
                x = range_val * np.cos(angle)
                y = range_val * np.sin(angle)
                self.obstacles.append((x, y))

    def plan_global_path(self):
        """Plan global path to goal"""
        if self.current_goal:
            # In a real implementation, this would use occupancy grid
            # For this example, we'll use a simple grid
            start = (50, 50)  # Starting position in grid coordinates
            path = self.global_planner.plan_path(start, self.current_goal)

            # Publish global plan
            path_msg = Path()
            path_msg.header.frame_id = 'map'

            for point in path:
                pose = PoseStamped()
                pose.pose.position.x = point[0]
                pose.pose.position.y = point[1]
                path_msg.poses.append(pose)

            self.global_plan_pub.publish(path_msg)

    def execute_local_plan(self):
        """Execute local planning and control"""
        if self.current_goal and self.obstacles:
            # Get current robot pose (simplified)
            current_pose = (50.0, 50.0, 0.0)  # x, y, theta
            current_vel = (0.0, 0.0)  # linear, angular

            # Plan local trajectory
            cmd_vel = self.local_planner.plan_local_path(
                current_pose, current_vel, self.current_goal, self.obstacles
            )

            # Publish velocity command
            twist_msg = Twist()
            twist_msg.linear.x = cmd_vel[0]
            twist_msg.angular.z = cmd_vel[1]
            self.cmd_vel_pub.publish(twist_msg)
```

## Planning Quality Assessment

### Plan Evaluation Metrics

```python
class PlanningQualityAssessor:
    def __init__(self):
        self.metrics_history = []

    def evaluate_path_quality(self, path: List[Tuple[float, float]],
                            start: Tuple[float, float],
                            goal: Tuple[float, float]) -> Dict[str, float]:
        """Evaluate quality of planned path"""
        if not path:
            return {
                'path_length': float('inf'),
                'smoothness': 0.0,
                'clearance': 0.0,
                'completeness': 0.0
            }

        # Path length
        path_length = 0.0
        for i in range(1, len(path)):
            path_length += np.linalg.norm(
                np.array(path[i]) - np.array(path[i-1])
            )

        # Smoothness (average turning angle)
        smoothness = 0.0
        if len(path) >= 3:
            angles = []
            for i in range(1, len(path) - 1):
                v1 = np.array(path[i]) - np.array(path[i-1])
                v2 = np.array(path[i+1]) - np.array(path[i])

                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    cos_angle = np.clip(cos_angle, -1, 1)  # Avoid numerical errors
                    angle = np.arccos(cos_angle)
                    angles.append(angle)

            smoothness = 1.0 / (1.0 + np.mean(angles)) if angles else 1.0

        # Completeness (distance to goal)
        final_dist = np.linalg.norm(np.array(path[-1]) - np.array(goal))
        completeness = 1.0 / (1.0 + final_dist)

        # Path efficiency (ratio of direct distance to path length)
        direct_distance = np.linalg.norm(np.array(goal) - np.array(start))
        efficiency = direct_distance / path_length if path_length > 0 else 0.0

        metrics = {
            'path_length': path_length,
            'smoothness': smoothness,
            'completeness': completeness,
            'efficiency': efficiency,
            'direct_distance': direct_distance
        }

        self.metrics_history.append(metrics)
        return metrics

    def assess_decision_quality(self, decision: str, context: Dict,
                              outcomes: List[Dict]) -> Dict[str, float]:
        """Assess quality of decisions"""
        # Calculate decision effectiveness
        if outcomes:
            success_count = sum(1 for outcome in outcomes if outcome.get('success', False))
            success_rate = success_count / len(outcomes)

            # Average reward
            avg_reward = np.mean([outcome.get('reward', 0) for outcome in outcomes])

            # Risk assessment
            reward_std = np.std([outcome.get('reward', 0) for outcome in outcomes])

            return {
                'success_rate': success_rate,
                'avg_reward': avg_reward,
                'reward_std': reward_std,
                'risk_level': 'high' if reward_std > 1.0 else 'low'
            }

        return {
            'success_rate': 0.0,
            'avg_reward': 0.0,
            'reward_std': 0.0,
            'risk_level': 'unknown'
        }

    def detect_planning_anomalies(self, current_metrics: Dict) -> List[str]:
        """Detect anomalies in planning performance"""
        anomalies = []

        if len(self.metrics_history) >= 5:
            recent_metrics = self.metrics_history[-5:]

            # Check for sudden changes in path efficiency
            recent_efficiency = [m['efficiency'] for m in recent_metrics]
            avg_efficiency = np.mean(recent_efficiency)
            current_efficiency = current_metrics['efficiency']

            if abs(current_efficiency - avg_efficiency) > 0.3:  # 30% change
                anomalies.append('path_efficiency_degradation')

        return anomalies
```

## Key Takeaways

- Planning algorithms enable robots to find optimal paths and sequences of actions
- Classical approaches (A*, Dijkstra) work well for known environments
- Sampling-based methods (RRT) handle complex configuration spaces
- Decision making under uncertainty requires probabilistic approaches (MDP, POMDP)
- Hierarchical planning decomposes complex tasks into manageable subtasks
- Real-time planning adapts to dynamic environments
- Quality assessment ensures reliable planning performance

## Next Steps

In the next chapter, we'll explore control systems and motor skills, learning how AI robot brains translate high-level plans into precise physical actions.