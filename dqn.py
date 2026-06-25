import numpy as np
import matplotlib.pyplot as plt

import random
from tqdm import tqdm
import gymnasium as gym
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(
            (state, action, reward, next_state, done)
        )

    def sample(self, batch_size):

        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:

    def __init__(self):

        self.env = gym.make("CartPole-v1")

        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n

        self.q_net = DQN(state_dim, action_dim)

        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(
            self.q_net.state_dict()
        )

        self.optimizer = optim.Adam(
            self.q_net.parameters(),
            lr=1e-3
        )

        self.buffer = ReplayBuffer()

        self.gamma = 0.99
        self.batch_size = 64

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        self.target_update = 20

    def select_action(self, state):

        if random.random() < self.epsilon:
            return self.env.action_space.sample()

        state = (
            torch.FloatTensor(state)
            .unsqueeze(0)
        )

        with torch.no_grad():
            q = self.q_net(state)

        return q.argmax().item()

    def train_step(self):

        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = (
            self.buffer.sample(
                self.batch_size
            )
        )

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)

        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)

        dones = torch.FloatTensor(dones)

        # Q(s,a)
        q_values = (
            self.q_net(states)
            .gather(
                1,
                actions.unsqueeze(1)
            )
            .squeeze()
        )

        # max_a' Q_target(s',a')
        with torch.no_grad():

            next_q = (
                self.target_net(next_states)
                .max(1)[0]
            )

            targets = (
                rewards
                + self.gamma
                * next_q
                * (1 - dones)
            )

        loss = nn.functional.mse_loss(
            q_values,
            targets
        )

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

    def train(self, episodes=400):

        rewards_history = []

        for ep in range(episodes):

            state, _ = self.env.reset()

            done = False

            total_reward = 0

            while not done:

                action = self.select_action(
                    state
                )

                next_state, reward, terminated, truncated, _ = (
                    self.env.step(action)
                )

                done = (
                    terminated
                    or truncated
                )

                self.buffer.push(
                    state,
                    action,
                    reward,
                    next_state,
                    done
                )

                self.train_step()

                state = next_state

                total_reward += reward

            rewards_history.append(
                total_reward
            )

            self.epsilon = max(
                self.epsilon_min,
                self.epsilon
                * self.epsilon_decay
            )

            if ep % self.target_update == 0:
                self.target_net.load_state_dict(
                    self.q_net.state_dict()
                )

            avg = np.mean(
                rewards_history[-20:]
            )

            print(
                f"Episode {ep:3d}"
                f" | reward={total_reward:4.0f}"
                f" | avg20={avg:.1f}"
                f" | eps={self.epsilon:.3f}"
            )

        return rewards_history


agent = DQNAgent()

# Train
rewards = agent.train()

plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Training Progress")
plt.show()

# Evaluate
env = gym.make(
    "CartPole-v1",
    render_mode="human"
)

state, _ = env.reset()

done = False

while not done:

    with torch.no_grad():

        action = (
            agent.q_net(
                torch.FloatTensor(
                    state
                ).unsqueeze(0)
            )
            .argmax()
            .item()
        )

    state, reward, terminated, truncated, _ = (
        env.step(action)
    )

    done = (
        terminated
        or truncated
    )

env.close()