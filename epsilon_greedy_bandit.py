import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

class EpsilonGreedyBandit:
    def __init__(self, k=10, epsilon=0.1):
        self.k = k
        self.epsilon = epsilon

        # Estimated action values
        self.Q = np.zeros(k)

        # Action counts
        self.N = np.zeros(k)

    def select_action(self):
        """
        Choose action using epsilon-greedy strategy.
        """
        if np.random.rand() < self.epsilon:
            # Explore
            return np.random.randint(self.k)
        else:
            # Exploit
            return np.argmax(self.Q)

    def update(self, action, reward):
        """
        Incremental update of action-value estimate.
        """
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]


class KArmedBandit:
    def __init__(self, k=10):
        self.k = k
        # True mean reward for each arm
        self.true_values = np.random.normal(0, 1, k)

    def pull(self, action):
        """
        Reward sampled from:
        R ~ N(true_value[action], 1)
        """
        return np.random.normal(self.true_values[action], 1)

def main():

    k = 10
    steps = 5000
    epsilons = [0.0, 0.01, 0.1]

    # Same environment for all runs
    env = KArmedBandit(k)
    results = {}

    for epsilon in epsilons:

        agent = EpsilonGreedyBandit(
            k=k,
            epsilon=epsilon
        )

        rewards = []

        for t in range(steps):

            action = agent.select_action()
            reward = env.pull(action)
            agent.update(action, reward)
            rewards.append(reward)

        # running average reward
        avg_rewards = np.cumsum(rewards) / (np.arange(steps) + 1)

        results[epsilon] = avg_rewards


    plt.figure(figsize=(10, 6))

    for epsilon in epsilons:
        plt.plot(
            results[epsilon],
            label=f"eps = {epsilon}"
        )

    plt.xlabel("Time step")
    plt.ylabel("Average reward")
    plt.title("Epsilon-Greedy k-Armed Bandit")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()