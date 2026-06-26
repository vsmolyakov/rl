import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
from collections import defaultdict

class TileCoder:
    def __init__(self, low, high, num_tilings=8, bins=(8, 8)):
        self.low = low
        self.high = high
        self.num_tilings = num_tilings
        self.bins = np.array(bins)

        self.tile_width = (high - low) / (self.bins - 1)

        # offset each tiling slightly
        self.offsets = [
            (i / num_tilings) * self.tile_width
            for i in range(num_tilings)
        ]

    def get_tiles(self, state):
        features = []

        for tiling, offset in enumerate(self.offsets):

            scaled = (
                (state - self.low + offset)
                / self.tile_width
            ).astype(int)

            scaled = np.clip(
                scaled,
                0,
                self.bins - 1
            )

            features.append(
                (
                    tiling,
                    scaled[0],
                    scaled[1]
                )
            )

        return features

class SarsaLambdaAgent:

    def __init__(
        self,
        n_actions,
        tile_coder,
        alpha=0.3,
        gamma=1.0,
        lam=0.9,
        epsilon=0.05
    ):
        self.n_actions = n_actions
        self.tc = tile_coder

        self.alpha = alpha / tile_coder.num_tilings
        self.gamma = gamma
        self.lam = lam
        self.epsilon = epsilon

        self.w = defaultdict(float)

    def q_value(self, state, action):

        return sum(
            self.w[(f, action)]
            for f in self.tc.get_tiles(state)
        )

    def choose_action(self, state):

        if np.random.rand() < self.epsilon:
            return np.random.randint(
                self.n_actions
            )

        qs = [
            self.q_value(state, a)
            for a in range(self.n_actions)
        ]

        return np.argmax(qs)

    def update(
        self,
        state,
        action,
        reward,
        next_state,
        next_action,
        done,
        traces
    ):

        q = self.q_value(state, action)

        if done:
            target = reward
        else:
            target = (
                reward
                + self.gamma
                * self.q_value(
                    next_state,
                    next_action
                )
            )

        delta = target - q

        # decay traces
        for k in list(traces.keys()):
            traces[k] *= (
                self.gamma
                * self.lam
            )

        # replacing traces
        for f in self.tc.get_tiles(state):
            traces[(f, action)] = 1.0

        # update weights
        for k in traces:
            self.w[k] += (
                self.alpha
                * delta
                * traces[k]
            )

        return traces


env = gym.make("MountainCar-v0")

tc = TileCoder(
    env.observation_space.low,
    env.observation_space.high,
    num_tilings=8,
    bins=(8, 8)
)

agent = SarsaLambdaAgent(
    n_actions=env.action_space.n,
    tile_coder=tc,
    alpha=0.5,
    gamma=1.0,
    lam=0.9,
    epsilon=0.05
)

episodes = 500
returns = []

for ep in range(episodes):

    state, _ = env.reset()

    traces = defaultdict(float)

    action = agent.choose_action(
        state
    )

    total_reward = 0

    done = False

    while not done:

        next_state, reward, terminated, truncated, _ = (
            env.step(action)
        )

        done = (
            terminated
            or truncated
        )

        total_reward += reward

        if not done:
            next_action = (
                agent.choose_action(
                    next_state
                )
            )
        else:
            next_action = None

        traces = agent.update(
            state,
            action,
            reward,
            next_state,
            next_action,
            done,
            traces
        )

        state = next_state
        action = next_action

    returns.append(
        total_reward
    )

    if (ep + 1) % 50 == 0:
        print(
            f"Episode {ep+1}, "
            f"avg return="
            f"{np.mean(returns[-50:]):.1f}"
        )

env.close()

window = 50
avg = np.convolve(
    returns,
    np.ones(window)
    / window,
    mode="valid"
)
plt.plot(avg, color='red', linewidth=2)
plt.xlabel("Episode")
plt.ylabel("Avg Return")
plt.title("SARSA(lambda) on MountainCar")
plt.show()