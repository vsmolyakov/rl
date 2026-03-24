import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np
import gymnasium as gym
from itertools import count
from collections import namedtuple

env = gym.make('CartPole-v1', render_mode='human')
env.reset(seed=42)
torch.manual_seed(42)

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.actor_fc = nn.Linear(128, action_size)
        self.critic_fc = nn.Linear(128, 1)
        self.drop = nn.Dropout(p=0.6)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = self.fc1(x)
        x = self.drop(x)
        x = F.relu(x)
        action_scores = self.actor_fc(x)
        state_value = self.critic_fc(x)
        return F.softmax(action_scores, dim=-1), state_value
    
model = ActorCritic(state_size=4, action_size=2)
optimizer = optim.Adam(model.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()

def select_action(state):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)

    m = Categorical(probs)
    action = m.sample()
    
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

    return action.item()

def finish_episode():
    R = 0
    gamma = 0.99
    saved_actions = model.saved_actions
    policy_losses = []
    value_losses = []
    returns = []
    for r in model.rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    
    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()
        
        #actor loss 
        policy_losses.append(-log_prob * advantage)

        #critic loss
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    loss.backward()
    optimizer.step()

    del model.rewards[:]
    del model.saved_actions[:]

def main():

    running_reward = 10

    for i_episode in count(1):

        state, _ = env.reset()
        ep_reward = 0
        for t in range(1, 10000):
            action = select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            model.rewards.append(reward)
            ep_reward += reward
            if terminated or truncated:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        finish_episode()

        if i_episode % 10 == 0:
            print(f'Episode {i_episode}\tLast reward: {ep_reward:.2f}\tAverage reward: {running_reward:.2f}')

        if running_reward > env.spec.reward_threshold:
            print(f"Solved! Running reward is now {running_reward} and "
                  f"the last episode runs to {t} time steps!")
            break

if __name__ == '__main__':
    main()