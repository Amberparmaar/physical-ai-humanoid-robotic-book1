---
sidebar_position: 5
title: Learning and Adaptation
---

# Learning and Adaptation

In this chapter, we'll explore how AI robot brains learn from experience and adapt to changing conditions. Learning and adaptation are fundamental capabilities that enable robots to improve their performance over time, handle novel situations, and operate effectively in dynamic environments.

## Understanding Learning in Robotics

Learning in robotics encompasses several key paradigms:

### 1. Supervised Learning
- Learning from labeled examples
- Used for perception tasks (object recognition, scene understanding)
- Requires large datasets of input-output pairs

### 2. Unsupervised Learning
- Discovering patterns in unlabeled data
- Used for clustering, dimensionality reduction
- Helps understand environment structure

### 3. Reinforcement Learning
- Learning through interaction with environment
- Maximizing cumulative rewards
- Essential for decision-making and control

### 4. Imitation Learning
- Learning by observing expert demonstrations
- Fast learning of complex behaviors
- Good initialization for reinforcement learning

## Supervised Learning for Robotics

### Deep Learning for Perception

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

class PerceptionNetwork(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(PerceptionNetwork, self).__init__()

        # Convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class SupervisedLearner:
    def __init__(self, model: nn.Module, learning_rate: float = 0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.train_losses = []
        self.val_accuracies = []

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0

        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss

    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model"""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        accuracy = 100 * correct / total
        self.val_accuracies.append(accuracy)
        return accuracy

    def predict(self, data):
        """Make predictions on new data"""
        self.model.eval()
        with torch.no_grad():
            if isinstance(data, np.ndarray):
                data = torch.FloatTensor(data).unsqueeze(0)
            data = data.to(self.device)
            outputs = self.model(data)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)

        return predicted_class.cpu().numpy()[0], probabilities.cpu().numpy()[0]
```

### Transfer Learning for Robotics

```python
import torchvision.models as models
from torch import nn

class TransferLearningRobot:
    def __init__(self, base_model_name: str = 'resnet18', num_classes: int = 10):
        # Load pre-trained model
        if base_model_name == 'resnet18':
            self.base_model = models.resnet18(pretrained=True)
            num_features = self.base_model.fc.in_features
        elif base_model_name == 'vgg16':
            self.base_model = models.vgg16(pretrained=True)
            num_features = self.base_model.classifier[6].in_features

        # Replace final classifier for robotics tasks
        self.base_model.fc = nn.Linear(num_features, num_features // 2)

        # Add robotics-specific layers
        self.robot_classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(num_features // 2, num_classes)
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = nn.Sequential(
            self.base_model,
            self.robot_classifier
        ).to(self.device)

        # Freeze early layers for transfer learning
        self.freeze_base_layers(freeze=True)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

    def freeze_base_layers(self, freeze: bool = True):
        """Freeze or unfreeze base model layers"""
        for param in self.base_model.parameters():
            param.requires_grad = not freeze

    def fine_tune(self, train_loader, num_epochs: int = 10):
        """Fine-tune the model on robotics data"""
        # First, train only the new classifier layers
        self.freeze_base_layers(freeze=True)
        self._train_epochs(train_loader, num_epochs // 2)

        # Then, unfreeze and train all layers with lower learning rate
        self.freeze_base_layers(freeze=False)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = 0.0001  # Lower learning rate for fine-tuning

        self._train_epochs(train_loader, num_epochs // 2)

    def _train_epochs(self, train_loader, num_epochs: int):
        """Train for specified number of epochs"""
        for epoch in range(num_epochs):
            total_loss = 0.0
            for data, targets in train_loader:
                data, targets = data.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
```

## Reinforcement Learning for Robotics

### Deep Q-Network (DQN) for Continuous Control

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, lr: float = 0.001,
                 gamma: float = 0.99, epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Neural networks
        self.q_network = DQN(state_dim, action_dim).to(self.device)
        self.target_network = DQN(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Experience replay
        self.memory = deque(maxlen=10000)
        self.batch_size = 32

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
            return random.randrange(self.action_dim)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())

    def replay(self):
        """Train the model on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def learn(self, env, episodes: int = 1000):
        """Learn to perform tasks in the environment"""
        scores = []

        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False

            while not done:
                action = self.act(state)
                next_state, reward, done, info = env.step(action)

                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                # Train on experiences
                self.replay()

                # Update target network periodically
                if episode % 100 == 0:
                    self.update_target_network()

            scores.append(total_reward)

            if episode % 100 == 0:
                avg_score = np.mean(scores[-100:])
                print(f"Episode {episode}, Average Score: {avg_score:.2f}, Epsilon: {self.epsilon:.3f}")

        return scores
```

### Actor-Critic Methods (DDPG for Continuous Actions)

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float, hidden_dim: int = 256):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q = torch.relu(self.l1(sa))
        q = torch.relu(self.l2(q))
        return self.l3(q)

class DDPGAgent:
    def __init__(self, state_dim: int, action_dim: int, max_action: float,
                 lr_actor: float = 1e-4, lr_critic: float = 1e-3, tau: float = 0.005):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.max_action = max_action
        self.tau = tau

    def select_action(self, state, noise_scale: float = 0.1):
        """Select action with added noise for exploration"""
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()

        # Add exploration noise
        noise = np.random.normal(0, noise_scale, size=action.shape)
        action = np.clip(action + noise, -self.max_action, self.max_action)

        return action

    def update(self, replay_buffer, batch_size: int = 100):
        """Update actor and critic networks"""
        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        not_done = torch.FloatTensor(not_done).to(self.device)

        # Compute target Q-value
        next_action = self.actor_target(next_state)
        target_Q = self.critic_target(next_state, next_action)
        target_Q = reward + (not_done * 0.99 * target_Q).detach()

        # Optimize Critic
        current_Q = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

class ReplayBuffer:
    def __init__(self, max_size: int = int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, 0))
        self.action = np.zeros((max_size, 0))
        self.next_state = np.zeros((max_size, 0))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

    def add(self, state, action, next_state, reward, done):
        if self.state.shape[1] == 0:  # Initialize with first sample
            self.state = np.zeros((self.max_size, state.shape[0]))
            self.action = np.zeros((self.max_size, action.shape[0]))
            self.next_state = np.zeros((self.max_size, next_state.shape[0]))

        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            self.state[ind],
            self.action[ind],
            self.next_state[ind],
            self.reward[ind],
            self.not_done[ind]
        )
```

## Imitation Learning

### Behavioral Cloning

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class BehavioralCloningNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(BehavioralCloningNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        return self.network(state)

class BehavioralCloningAgent:
    def __init__(self, state_dim: int, action_dim: int, lr: float = 1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.network = BehavioralCloningNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.demonstrations = []
        self.epochs = 100

    def add_demonstration(self, state: np.ndarray, action: np.ndarray):
        """Add a state-action pair from expert demonstration"""
        self.demonstrations.append((state, action))

    def train(self):
        """Train the network to mimic expert behavior"""
        if not self.demonstrations:
            print("No demonstrations available for training")
            return

        states = torch.FloatTensor([d[0] for d in self.demonstrations]).to(self.device)
        actions = torch.FloatTensor([d[1] for d in self.demonstrations]).to(self.device)

        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            predicted_actions = self.network(states)
            loss = self.criterion(predicted_actions, actions)
            loss.backward()
            self.optimizer.step()

            if epoch % 20 == 0:
                print(f"BC Epoch {epoch}, Loss: {loss.item():.4f}")

    def predict(self, state: np.ndarray) -> np.ndarray:
        """Predict action for given state"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.network(state_tensor).cpu().data.numpy()[0]
        return action

    def collect_demonstrations_from_robot(self, robot_env, expert_policy, num_demos: int = 100):
        """Collect demonstrations by running expert policy"""
        for demo_idx in range(num_demos):
            state = robot_env.reset()
            done = False

            while not done:
                action = expert_policy(state)
                next_state, reward, done, info = robot_env.step(action)

                self.add_demonstration(state, action)
                state = next_state
```

### Generative Adversarial Imitation Learning (GAIL)

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Discriminator(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(Discriminator, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, state, action):
        inputs = torch.cat([state, action], dim=1)
        return self.network(inputs)

class GAILAgent:
    def __init__(self, state_dim: int, action_dim: int,
                 actor_lr: float = 1e-4, discriminator_lr: float = 1e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Actor network (policy)
        self.actor = Actor(state_dim, action_dim, max_action=1.0).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Discriminator network
        self.discriminator = Discriminator(state_dim, action_dim).to(self.device)
        self.discriminator_optimizer = optim.Adam(
            self.discriminator.parameters(), lr=discriminator_lr
        )

        self.expert_buffer = ReplayBuffer()
        self.agent_buffer = ReplayBuffer()

        self.gamma = 0.99
        self.entropy_coef = 0.01

    def update_discriminator(self, expert_batch, agent_batch):
        """Update discriminator to distinguish expert vs agent trajectories"""
        expert_states, expert_actions = expert_batch
        agent_states, agent_actions = agent_batch

        expert_states = torch.FloatTensor(expert_states).to(self.device)
        expert_actions = torch.FloatTensor(expert_actions).to(self.device)
        agent_states = torch.FloatTensor(agent_states).to(self.device)
        agent_actions = torch.FloatTensor(agent_actions).to(self.device)

        # Discriminator loss: minimize -log(D(s,a)) for expert and -log(1-D(s,a)) for agent
        expert_logit = self.discriminator(expert_states, expert_actions)
        agent_logit = self.discriminator(agent_states, agent_actions)

        expert_loss = -torch.log(expert_logit + 1e-8).mean()
        agent_loss = -torch.log(1 - agent_logit + 1e-8).mean()

        discriminator_loss = expert_loss + agent_loss

        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.step()

        return discriminator_loss.item()

    def get_reward_from_discriminator(self, state, action):
        """Get reward from discriminator (1 - D(s,a))"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)

        prob_expert = self.discriminator(state_tensor, action_tensor)
        # Use log-prob as reward for gradient computation
        reward = -torch.log(1 - prob_expert + 1e-8).item()

        return reward

    def update_actor(self, states, actions, rewards, log_probs):
        """Update actor using policy gradient with discriminator-based rewards"""
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        log_probs = torch.FloatTensor(log_probs).to(self.device)

        # Calculate policy gradient
        predicted_actions = self.actor(states)
        log_probs_new = torch.log_softmax(predicted_actions, dim=1) * actions

        # Policy gradient loss
        actor_loss = -(log_probs_new * rewards.unsqueeze(1)).mean() - self.entropy_coef * torch.mean(log_probs_new)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item()
```

## Online Learning and Adaptation

### Incremental Learning with Elastic Weight Consolidation

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class EWCNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(EWCNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ElasticWeightConsolidation:
    def __init__(self, model: nn.Module, lambda_reg: float = 1000.0):
        self.model = model
        self.lambda_reg = lambda_reg

        # Store important weights and their optimal values
        self.fisher_information = {}
        self.optimal_params = {}

    def compute_fisher_information(self, dataloader, device):
        """Compute Fisher Information Matrix for important weights"""
        self.model.eval()
        self.fisher_information = {}

        # Initialize Fisher Information for all parameters
        for name, param in self.model.named_parameters():
            self.fisher_information[name] = torch.zeros_like(param)

        # Compute Fisher Information
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)

            self.model.zero_grad()
            output = self.model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    self.fisher_information[name] += param.grad.data ** 2

        # Normalize by dataset size
        for name in self.fisher_information:
            self.fisher_information[name] /= len(dataloader.dataset)

        # Store optimal parameters
        self.optimal_params = {name: param.clone() for name, param in self.model.named_parameters()}

    def ewc_loss(self):
        """Compute EWC regularization loss"""
        if not self.optimal_params:
            return torch.tensor(0.0)

        loss = 0.0
        for name, param in self.model.named_parameters():
            if name in self.optimal_params:
                fisher = self.fisher_information[name]
                opt_param = self.optimal_params[name]
                loss += (fisher * (param - opt_param) ** 2).sum()

        return self.lambda_reg * loss

    def update_model(self, new_data_loader, device, epochs: int = 10):
        """Update model on new task while preserving old knowledge"""
        optimizer = optim.Adam(self.model.parameters())

        for epoch in range(epochs):
            for data, target in new_data_loader:
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()

                output = self.model(data)
                task_loss = nn.CrossEntropyLoss()(output, target)

                # Add EWC regularization
                ewc_loss = self.ewc_loss()
                total_loss = task_loss + ewc_loss

                total_loss.backward()
                optimizer.step()
```

### Bayesian Neural Networks for Uncertainty Quantification

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BayesianLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, prior_std: float = 1.0):
        super(BayesianLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std

        # Learnable parameters for variational distribution
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_rho = nn.Parameter(torch.randn(out_features, in_features) * 0.1)

        self.bias_mu = nn.Parameter(torch.randn(out_features) * 0.1)
        self.bias_rho = nn.Parameter(torch.randn(out_features) * 0.1)

        # Prior parameters
        self.weight_prior = torch.distributions.Normal(0, prior_std)
        self.bias_prior = torch.distributions.Normal(0, prior_std)

    def forward(self, x):
        # Compute variational parameters
        weight_std = torch.log1p(torch.exp(self.weight_rho))
        bias_std = torch.log1p(torch.exp(self.bias_rho))

        # Sample weights and biases
        weight_eps = torch.randn_like(self.weight_mu)
        bias_eps = torch.randn_like(self.bias_mu)

        weight = self.weight_mu + weight_std * weight_eps
        bias = self.bias_mu + bias_std * bias_eps

        # Compute KL divergence
        weight_posterior = torch.distributions.Normal(self.weight_mu, weight_std)
        bias_posterior = torch.distributions.Normal(self.bias_mu, bias_std)

        weight_kl = torch.distributions.kl_divergence(weight_posterior, self.weight_prior).sum()
        bias_kl = torch.distributions.kl_divergence(bias_posterior, self.bias_prior).sum()

        self.kl_divergence = weight_kl + bias_kl

        return F.linear(x, weight, bias)

class BayesianNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(BayesianNetwork, self).__init__()

        self.bayesian1 = BayesianLinear(input_dim, hidden_dim)
        self.bayesian2 = BayesianLinear(hidden_dim, hidden_dim)
        self.bayesian3 = BayesianLinear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.bayesian1(x))
        x = torch.relu(self.bayesian2(x))
        return self.bayesian3(x)

    def get_kl_divergence(self):
        """Get total KL divergence for all layers"""
        kl_total = 0
        for module in self.modules():
            if hasattr(module, 'kl_divergence'):
                kl_total += module.kl_divergence
        return kl_total

class BayesianLearner:
    def __init__(self, input_dim: int, output_dim: int, lr: float = 0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BayesianNetwork(input_dim, 128, output_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train_step(self, data, targets, beta: float = 1.0):
        """Training step with Bayesian regularization"""
        data, targets = data.to(self.device), targets.to(self.device)

        self.optimizer.zero_grad()

        outputs = self.model(data)
        likelihood_loss = nn.MSELoss()(outputs, targets)

        # Add KL divergence as regularization
        kl_divergence = self.model.get_kl_divergence()
        total_loss = likelihood_loss + beta * kl_divergence

        total_loss.backward()
        self.optimizer.step()

        return total_loss.item(), likelihood_loss.item(), kl_divergence.item()

    def predict_with_uncertainty(self, x, num_samples: int = 100):
        """Get predictions with uncertainty estimates"""
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for _ in range(num_samples):
                output = self.model(x.to(self.device))
                predictions.append(output.cpu().numpy())

        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)

        return mean_pred, std_pred
```

## Meta-Learning and Few-Shot Adaptation

### Model-Agnostic Meta-Learning (MAML)

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class MAMLNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        super(MAMLNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)

class MAML:
    def __init__(self, model: nn.Module, inner_lr: float = 0.01, outer_lr: float = 0.001):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr

        self.optimizer = optim.Adam(model.parameters(), lr=outer_lr)
        self.criterion = nn.MSELoss()

    def clone_model(self):
        """Create a copy of the model with detached parameters"""
        cloned_model = MAMLNetwork(self.model.network[0].in_features,
                                  self.model.network[-1].out_features)
        cloned_model.load_state_dict(self.model.state_dict())
        return cloned_model

    def inner_update(self, model, support_data, support_labels):
        """Perform inner loop update (adaptation to new task)"""
        model.train()

        # Detach gradients for inner loop
        for param in model.parameters():
            param.requires_grad_(True)

        optimizer = optim.SGD(model.parameters(), lr=self.inner_lr)

        # Adapt to support set
        for _ in range(5):  # Inner loop steps
            optimizer.zero_grad()
            outputs = model(support_data)
            loss = self.criterion(outputs, support_labels)
            loss.backward()
            optimizer.step()

        return model

    def forward(self, batch_data):
        """Forward pass for meta-learning"""
        total_loss = 0

        for task_data in batch_data:
            support_data, support_labels = task_data['support']
            query_data, query_labels = task_data['query']

            # Clone model for this task
            task_model = self.clone_model()

            # Inner loop: adapt to task
            adapted_model = self.inner_update(task_model, support_data, support_labels)

            # Outer loop: evaluate on query set
            query_outputs = adapted_model(query_data)
            task_loss = self.criterion(query_outputs, query_labels)

            total_loss += task_loss

        return total_loss / len(batch_data)

    def train_step(self, batch_data):
        """Single training step"""
        self.optimizer.zero_grad()

        loss = self.forward(batch_data)
        loss.backward()

        self.optimizer.step()

        return loss.item()
```

## Continual Learning for Robotics

### Progressive Neural Networks

```python
import torch
import torch.nn as nn
import torch.optim as optim

class ProgressiveLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, lateral_connections: list = None):
        super(ProgressiveLayer, self).__init__()

        self.main_layer = nn.Linear(input_dim, output_dim)
        self.lateral_connections = lateral_connections or []

        # Lateral connections from previous columns
        self.lateral_weights = nn.ModuleList()
        for prev_output_dim in lateral_connections:
            self.lateral_weights.append(nn.Linear(prev_output_dim, output_dim))

    def forward(self, x, lateral_inputs=None):
        main_output = torch.relu(self.main_layer(x))

        lateral_output = 0
        if lateral_inputs is not None:
            for i, lateral_input in enumerate(lateral_inputs):
                lateral_output += self.lateral_weights[i](lateral_input)

        return main_output + lateral_output

class ProgressiveNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_tasks: int, hidden_dim: int = 128):
        super(ProgressiveNetwork, self).__init__()

        self.num_tasks = num_tasks
        self.hidden_dim = hidden_dim

        # Task-specific columns
        self.columns = nn.ModuleList()

        for task_idx in range(num_tasks):
            # Determine lateral connections for this column
            lateral_dims = [hidden_dim] * task_idx if task_idx > 0 else []

            column = nn.Sequential(
                ProgressiveLayer(input_dim if task_idx == 0 else hidden_dim, hidden_dim, lateral_dims),
                ProgressiveLayer(hidden_dim, hidden_dim, [hidden_dim] * (task_idx + 1) if task_idx > 0 else []),
                ProgressiveLayer(hidden_dim, output_dim, [hidden_dim] * (task_idx + 1) if task_idx > 0 else [])
            )

            self.columns.append(column)

    def forward(self, x, task_idx: int = 0):
        column = self.columns[task_idx]

        # For progressive networks, we need to track lateral connections
        # This is a simplified implementation
        output = column[0](x)
        output = column[1](output)
        output = column[2](output)

        return output

class ProgressiveLearner:
    def __init__(self, input_dim: int, output_dim: int, num_tasks: int):
        self.model = ProgressiveNetwork(input_dim, output_dim, num_tasks)
        self.optimizers = [optim.Adam(list(self.model.columns[i].parameters()), lr=0.001)
                          for i in range(num_tasks)]
        self.criterion = nn.MSELoss()

    def train_task(self, task_idx: int, train_loader, epochs: int = 100):
        """Train a specific task column"""
        optimizer = self.optimizers[task_idx]
        column = self.model.columns[task_idx]

        for epoch in range(epochs):
            total_loss = 0
            for data, targets in train_loader:
                optimizer.zero_grad()

                outputs = column(data)
                loss = self.criterion(outputs, targets)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if epoch % 20 == 0:
                print(f"Task {task_idx}, Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}")
```

## Learning from Demonstration and Human Interaction

### Learning from Human Corrections

```python
class LearningFromCorrection:
    def __init__(self, base_policy, learning_rate: float = 0.1):
        self.base_policy = base_policy
        self.learning_rate = learning_rate
        self.correction_buffer = []
        self.policy_adjustments = {}

    def receive_correction(self, state: np.ndarray, action_taken: np.ndarray,
                          corrected_action: np.ndarray, reward: float = 0.0):
        """Receive a human correction and learn from it"""
        correction = {
            'state': state,
            'original_action': action_taken,
            'corrected_action': corrected_action,
            'reward': reward
        }

        self.correction_buffer.append(correction)

        # Update policy based on correction
        self._update_policy_from_correction(correction)

    def _update_policy_from_correction(self, correction: dict):
        """Update policy based on a single correction"""
        state = correction['state']
        original_action = correction['original_action']
        corrected_action = correction['corrected_action']

        # Calculate correction direction
        correction_direction = corrected_action - original_action

        # Update policy parameters (simplified - in practice, this would depend on policy type)
        if hasattr(self.base_policy, 'adjust_for_state'):
            self.base_policy.adjust_for_state(state, correction_direction * self.learning_rate)

    def get_corrected_action(self, state: np.ndarray) -> np.ndarray:
        """Get action from policy with learned corrections"""
        base_action = self.base_policy.get_action(state)

        # Apply learned corrections based on similar states
        correction = self._find_similar_correction(state)

        if correction is not None:
            return base_action + correction
        else:
            return base_action

    def _find_similar_correction(self, state: np.ndarray) -> np.ndarray:
        """Find similar states in correction buffer and return average correction"""
        if not self.correction_buffer:
            return None

        # Find most similar state (using simple distance measure)
        min_distance = float('inf')
        best_correction = None

        for correction in self.correction_buffer:
            distance = np.linalg.norm(state - correction['state'])
            if distance < min_distance and distance < 0.1:  # Threshold for similarity
                min_distance = distance
                best_correction = correction['corrected_action'] - correction['original_action']

        return best_correction if best_correction is not None else None
```

## NVIDIA Isaac Learning Integration

### Isaac ROS Learning Components

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import numpy as np

class IsaacLearningNode(Node):
    def __init__(self):
        super().__init__('isaac_learning_node')

        # Publishers and subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)
        self.goal_sub = self.create_subscription(
            PoseStamped, '/move_base_simple/goal', self.goal_callback, 10)
        self.path_sub = self.create_subscription(
            Path, '/plan', self.path_callback, 10)

        # Learning components
        self.supervised_learner = None
        self.rl_agent = None
        self.demonstration_buffer = []

        # Robot state
        self.current_positions = {}
        self.current_velocities = {}
        self.current_goal = None
        self.current_path = []

        # Learning parameters
        self.learning_enabled = True
        self.demonstration_mode = False

        # Setup learning components
        self.setup_learning_components()

    def setup_learning_components(self):
        """Initialize learning components"""
        # Initialize supervised learning for perception
        perception_model = PerceptionNetwork(num_classes=10)
        self.supervised_learner = SupervisedLearner(perception_model)

        # Initialize reinforcement learning for navigation
        # State: [x, y, theta, goal_x, goal_y, obstacle_distances...]
        # Action: [linear_vel, angular_vel]
        self.rl_agent = DDPGAgent(state_dim=20, action_dim=2, max_action=1.0)

    def joint_state_callback(self, msg: JointState):
        """Update joint state"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_positions[name] = msg.position[i]
            if i < len(msg.velocity):
                self.current_velocities[name] = msg.velocity[i]

    def goal_callback(self, msg: PoseStamped):
        """Update goal"""
        self.current_goal = np.array([msg.pose.position.x, msg.pose.position.y])

    def path_callback(self, msg: Path):
        """Update planned path"""
        self.current_path = []
        for pose in msg.poses:
            self.current_path.append([
                pose.pose.position.x,
                pose.pose.position.y
            ])

    def collect_demonstration(self, state: np.ndarray, action: np.ndarray):
        """Collect demonstration for learning"""
        if self.demonstration_mode:
            self.demonstration_buffer.append((state, action))

            # Train behavioral cloning if enough demonstrations
            if len(self.demonstration_buffer) >= 100:
                self.train_from_demonstrations()

    def train_from_demonstrations(self):
        """Train from collected demonstrations"""
        if len(self.demonstration_buffer) == 0:
            return

        # Train behavioral cloning agent
        bc_agent = BehavioralCloningAgent(
            state_dim=len(self.demonstration_buffer[0][0]),
            action_dim=len(self.demonstration_buffer[0][1])
        )

        for state, action in self.demonstration_buffer:
            bc_agent.add_demonstration(state, action)

        bc_agent.train()
        print(f"Trained from {len(self.demonstration_buffer)} demonstrations")

        # Clear buffer for next batch
        self.demonstration_buffer = []

    def get_robot_state(self) -> np.ndarray:
        """Get current robot state vector"""
        # Combine joint positions, velocities, and goal information
        joint_positions = list(self.current_positions.values())
        joint_velocities = list(self.current_velocities.values())

        if self.current_goal is not None:
            goal_info = [self.current_goal[0], self.current_goal[1]]
        else:
            goal_info = [0.0, 0.0]

        state = np.concatenate([
            joint_positions,
            joint_velocities,
            goal_info
        ])

        return state

    def execute_learning_step(self):
        """Execute a step of the learning process"""
        if not self.learning_enabled:
            return

        # Get current state
        current_state = self.get_robot_state()

        # Get action from current policy
        if self.rl_agent:
            action = self.rl_agent.select_action(current_state)
        else:
            action = np.zeros(2)  # Default action

        # In a real system, you would execute this action
        # and observe the resulting state and reward
        # For simulation purposes, we'll just return
        pass

    def enable_demonstration_mode(self):
        """Enable demonstration collection mode"""
        self.demonstration_mode = True
        self.get_logger().info("Demonstration mode enabled")

    def disable_demonstration_mode(self):
        """Disable demonstration collection mode"""
        self.demonstration_mode = False
        self.get_logger().info("Demonstration mode disabled")

    def save_model(self, filepath: str):
        """Save the learned model"""
        if self.supervised_learner:
            torch.save(self.supervised_learner.model.state_dict(), filepath)
            self.get_logger().info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load a pre-trained model"""
        if self.supervised_learner:
            self.supervised_learner.model.load_state_dict(torch.load(filepath))
            self.get_logger().info(f"Model loaded from {filepath}")
```

## Learning Quality Assessment

### Learning Performance Metrics

```python
class LearningQualityAssessor:
    def __init__(self):
        self.performance_history = []
        self.learning_curves = {}
        self.transfer_efficiency = {}
        self.catastrophic_forgetting = {}

    def assess_learning_progress(self, task_performance: Dict[str, float],
                               learning_curve: List[float]) -> Dict[str, float]:
        """Assess learning progress and quality"""
        if not learning_curve:
            return {'learning_rate': 0.0, 'convergence': False, 'stability': 0.0}

        # Calculate learning rate (slope of early learning)
        early_phase = min(10, len(learning_curve) // 4)
        if early_phase >= 2:
            early_learning_rate = (learning_curve[early_phase-1] - learning_curve[0]) / early_phase
        else:
            early_learning_rate = 0.0

        # Calculate convergence (how well performance stabilizes)
        late_phase = learning_curve[-min(10, len(learning_curve)):]
        if len(late_phase) > 1:
            convergence = 1.0 / (1.0 + np.std(late_phase))  # Lower std = better convergence
        else:
            convergence = 0.0

        # Calculate stability (variance over time)
        stability = 1.0 / (1.0 + np.std(learning_curve)) if len(learning_curve) > 1 else 1.0

        # Calculate final performance improvement
        if len(learning_curve) > 1:
            improvement = learning_curve[-1] - learning_curve[0]
        else:
            improvement = 0.0

        metrics = {
            'learning_rate': early_learning_rate,
            'convergence': convergence > 0.8,
            'stability': stability,
            'improvement': improvement,
            'final_performance': learning_curve[-1] if learning_curve else 0.0,
            'asymptotic_performance': np.mean(late_phase) if late_phase else 0.0
        }

        self.performance_history.append(metrics)
        return metrics

    def assess_transfer_learning(self, source_task: str, target_task: str,
                               performance_before: float, performance_after: float) -> float:
        """Assess transfer learning effectiveness"""
        if performance_before == 0:
            # If no prior performance, transfer effectiveness is based on absolute performance
            transfer_efficiency = performance_after
        else:
            # Calculate improvement ratio
            improvement = (performance_after - performance_before) / performance_before
            transfer_efficiency = max(0, improvement)  # Clamp to [0, inf)

        self.transfer_efficiency[(source_task, target_task)] = transfer_efficiency
        return transfer_efficiency

    def detect_catastrophic_forgetting(self, task: str, old_performance: float,
                                     new_performance: float, threshold: float = 0.1) -> bool:
        """Detect catastrophic forgetting in continual learning"""
        if old_performance > 0:
            performance_drop = (old_performance - new_performance) / old_performance
            forgetting_detected = performance_drop > threshold
        else:
            forgetting_detected = False

        if task not in self.catastrophic_forgetting:
            self.catastrophic_forgetting[task] = []
        self.catastrophic_forgetting[task].append({
            'old_performance': old_performance,
            'new_performance': new_performance,
            'drop_ratio': performance_drop if old_performance > 0 else 0,
            'forgetting_detected': forgetting_detected
        })

        return forgetting_detected

    def assess_sample_efficiency(self, samples_used: int, performance_achieved: float,
                               baseline_samples: int, baseline_performance: float) -> float:
        """Assess how efficiently the learning algorithm uses samples"""
        if baseline_samples == 0 or baseline_performance == 0:
            return 1.0  # Cannot compute efficiency

        # Calculate sample efficiency ratio
        efficiency_ratio = (baseline_samples / samples_used) * (performance_achieved / baseline_performance)
        return efficiency_ratio

    def get_learning_insights(self) -> Dict[str, Any]:
        """Get comprehensive insights about learning process"""
        if not self.performance_history:
            return {'message': 'No learning data available'}

        recent_metrics = self.performance_history[-min(5, len(self.performance_history)):]

        insights = {
            'average_learning_rate': np.mean([m['learning_rate'] for m in recent_metrics]),
            'convergence_rate': np.mean([m['convergence'] for m in recent_metrics]),
            'stability_trend': 'improving' if len(recent_metrics) > 1 and
                              recent_metrics[-1]['stability'] > recent_metrics[0]['stability'] else 'declining',
            'transfer_success_rate': len([k for k, v in self.transfer_efficiency.items() if v > 0.5]) /
                                    max(1, len(self.transfer_efficiency)),
            'forgetting_instances': sum(1 for task_forgetting in self.catastrophic_forgetting.values()
                                      for instance in task_forgetting if instance['forgetting_detected']),
            'sample_efficiency_score': np.mean([v for v in self.transfer_efficiency.values()]) if self.transfer_efficiency else 0.0
        }

        return insights
```

## Key Takeaways

- Learning enables robots to improve performance through experience
- Supervised learning is effective for perception and classification tasks
- Reinforcement learning enables robots to learn optimal behaviors through interaction
- Imitation learning allows fast skill acquisition from expert demonstrations
- Online learning adapts to changing conditions in real-time
- Continual learning prevents catastrophic forgetting when learning new tasks
- Quality assessment ensures learning algorithms perform effectively

## Next Steps

In the next chapter, we'll explore human-robot interaction systems, learning how AI robot brains enable natural and effective collaboration between humans and robots.