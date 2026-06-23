import numpy as np
import gymnasium as gym
from tqdm import tqdm
from collections import defaultdict

class OffPolicyMonteCarloBlackjack:
    def __init__(self, gamma=1.0):
        self.gamma = gamma

        # Action-value estimates
        self.Q = defaultdict(lambda: np.zeros(2))

        # Cumulative importance weights
        self.C = defaultdict(lambda: np.zeros(2))

        # Target policy (greedy)
        self.policy = defaultdict(lambda: 0)

    def behavior_policy(self, state):
        """
        Uniform random behavior policy.
        Returns action probabilities.
        """
        return np.array([0.5, 0.5])

    def select_behavior_action(self, state):
        probs = self.behavior_policy(state)
        return np.random.choice([0, 1], p=probs)

    def select_target_action(self, state):
        return np.argmax(self.Q[state])

    def generate_episode(self, env):
        episode = []

        state, _ = env.reset()
        done = False

        while not done:
            action = self.select_behavior_action(state)

            next_state, reward, terminated, truncated, _ = env.step(
                action
            )

            done = terminated or truncated

            episode.append(
                (state, action, reward)
            )

            state = next_state

        return episode

    def train(self, env, num_episodes=500000):

        for episode_num in tqdm(range(num_episodes), desc="Training"):

            episode = self.generate_episode(env)

            G = 0
            W = 1

            # Traverse backwards
            for t in reversed(range(len(episode))):

                state, action, reward = episode[t]

                G = self.gamma * G + reward

                # update cumulative importance weight
                self.C[state][action] += W

                # update action-value estimate using weighted importance sampling
                self.Q[state][action] += (W / self.C[state][action]) * (G - self.Q[state][action])

                # Improve target policy
                self.policy[state] = np.argmax(self.Q[state])

                # Stop if behavior deviates to avoid having W = 0
                if action != self.policy[state]:
                    break

                behavior_prob = (
                    self.behavior_policy(state)[action]
                )

                #update importance weight:
                #pi(a|s) / b(a|s) = 1 / b(a|s) since pi is deterministic
                W *= 1 / behavior_prob

    def play(self, env, episodes=10):

        for ep in range(episodes):

            state, _ = env.reset()
            done = False

            total_reward = 0

            while not done:

                action = self.select_target_action(
                    state
                )

                state, reward, terminated, truncated, _ = (
                    env.step(action)
                )

                done = terminated or truncated

                total_reward += reward

            print(
                f"Episode {ep+1}: reward={total_reward}"
            )


env = gym.make("Blackjack-v1")

agent = OffPolicyMonteCarloBlackjack()

agent.train(
    env,
    num_episodes=300000
)

print("\nExample learned states:")
print("(player sum, dealer showing, usable ace) -> action")

sample_states = [
    (20, 10, False),
    (18, 6, False),
    (13, 2, True),
]

for s in sample_states:
    action = agent.select_target_action(s)

    action_name = (
        "STICK" if action == 0 else "HIT"
    )

    print(
        f"{s}: {action_name}"
    )

print("\nPlaying...")

agent.play(env)