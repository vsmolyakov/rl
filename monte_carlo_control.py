import numpy as np
import gymnasium as gym
from collections import defaultdict

env = gym.make(
    "FrozenLake-v1",
    is_slippery=True
)

n_actions = env.action_space.n

def epsilon_greedy(Q, state, epsilon):

    if np.random.rand() < epsilon:
        return env.action_space.sample()

    return np.argmax(Q[state])

def generate_episode(Q, epsilon):

    episode = []

    state, _ = env.reset()

    done = False

    while not done:

        action = epsilon_greedy(
            Q,
            state,
            epsilon
        )

        next_state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        episode.append(
            (state, action, reward)
        )

        state = next_state

    return episode

def monte_carlo_control(
    episodes=10000,
    gamma=0.99,
    epsilon=0.2
):

    Q = defaultdict(lambda: np.zeros(n_actions))

    returns = defaultdict(list)

    rewards = []

    for ep in range(episodes):

        episode = generate_episode(
            Q,
            epsilon
        )

        G = 0
        visited = set()

        for t in reversed(
            range(len(episode))
        ):

            state, action, reward = episode[t]

            G = reward + gamma * G

            if (state, action) not in visited:

                visited.add(
                    (state, action)
                )

                returns[
                    (state, action)
                ].append(G)

                Q[state][action] = np.mean(
                    returns[
                        (state, action)
                    ]
                )

        rewards.append(
            sum(r for _, _, r in episode)
        )

        # decay exploration
        epsilon = max(
            0.01,
            epsilon * 0.99995
        )

    return Q, rewards


def extract_policy(Q):

    policy = {}

    for s in range(env.observation_space.n):

        policy[s] = np.argmax(
            Q[s]
        )

    return policy


Q, rewards = monte_carlo_control(
    episodes=10000
)

policy = extract_policy(Q)

action_names = [
    "←",
    "↓",
    "→",
    "↑"
]

print("\nLearned Policy:\n")

for s in range(16):

    print(
        action_names[
            policy[s]
        ],
        end=" "
    )

    if (s + 1) % 4 == 0:
        print()

print("\nAverage reward:")
print(np.mean(rewards[-5000:]))