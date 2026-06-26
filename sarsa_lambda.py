import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym

class TileCoder:

    def __init__(
        self,
        low,
        high,
        bins=(8, 8),
        num_tilings=8
    ):

        self.low = low
        self.high = high

        self.bins = np.array(bins)
        self.num_tilings = num_tilings

        self.tile_width = (high - low) / (self.bins - 1)

        self.offsets = [
            self.tile_width
            * i
            / num_tilings
            for i in range(num_tilings)
        ]

        self.tiles_per_tiling = (bins[0] * bins[1])
        self.num_features = (self.tiles_per_tiling * num_tilings)

    def encode(self, state):

        features = []

        for t in range(self.num_tilings):

            shifted = (state + self.offsets[t])
            coords = ((shifted - self.low) / self.tile_width).astype(int)

            coords = np.clip(
                coords,
                0,
                self.bins - 1
            )

            idx = (
                t
                * self.tiles_per_tiling
                + coords[0]
                * self.bins[1]
                + coords[1]
            )

            features.append(idx)

        return np.array(features)


class SARSALambda:

    def __init__(
        self,
        num_actions,
        num_features,
        alpha=0.3,
        gamma=0.99,
        lam=0.9,
        epsilon=0.1
    ):

        self.num_actions = num_actions
        self.num_features = num_features

        self.alpha = (alpha/8)

        self.gamma = gamma
        self.lam = lam
        self.epsilon = epsilon

        self.w = np.zeros((num_actions, num_features))

    def q(self, features):
        return (self.w[:,features].sum(axis=1))

    def select_action(
        self,
        features
    ):

        if (np.random.rand() < self.epsilon):
            return np.random.randint(self.num_actions)

        return np.argmax(self.q(features))

    def update(
        self,
        features,
        action,
        reward,
        next_features,
        next_action,
        done,
        eligibility
    ):

        current_q = (self.q(features)[action])

        if done:
            target = reward
        else:
            target = (
                reward
                + self.gamma
                * self.q(
                    next_features
                )[next_action]
            )

        delta = (target - current_q)

        eligibility *= (self.gamma * self.lam)

        eligibility[action, features] += 1

        self.w += (self.alpha * delta * eligibility)


env = gym.make(
    "MountainCar-v0",
)

coder = TileCoder(
    env.observation_space.low,
    env.observation_space.high
)

agent = SARSALambda(
    num_actions=env.action_space.n,
    num_features=coder.num_features,
    alpha=0.4,
    gamma=0.99,
    lam=0.9,
    epsilon=0.1
)

episodes = 500
returns = []

for episode in range(episodes):

    state, _ = env.reset()

    eligibility = np.zeros(
        (
            env.action_space.n,
            coder.num_features
        )
    )

    features = (coder.encode(state))

    action = (agent.select_action(features))

    total_reward = 0
    done = False

    while not done:

        (next_state, reward, terminated, truncated, _) = env.step(action)

        done = (terminated or truncated)

        if not done:

            next_features = (
                coder.encode(
                    next_state
                )
            )

            next_action = (
                agent
                .select_action(
                    next_features
                )
            )

        else:

            next_features = None
            next_action = None

        agent.update(
            features,
            action,
            reward,
            next_features,
            next_action,
            done,
            eligibility
        )

        state = next_state
        features = next_features
        action = next_action

        total_reward += reward

    returns.append(total_reward)

    if episode % 50 == 0:
        print(
            f"Episode "
            f"{episode}, "
            f"return="
            f"{total_reward}"
        )


env.close()


window = 20

avg = np.convolve(
    returns,
    np.ones(window)
    / window,
    mode="valid"
)

plt.plot(avg)
plt.xlabel("Episode")
plt.ylabel("Average Return")
plt.title("SARSA(lambda) - MountainCar")
plt.show()