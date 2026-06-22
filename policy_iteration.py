import numpy as np
import gymnasium as gym

def policy_evaluation(env, policy, gamma=0.99, theta=1e-8):

    n_states = env.observation_space.n
    V = np.zeros(n_states)

    while True:

        delta = 0

        for s in range(n_states):

            old = V[s]

            action = policy[s]

            value = 0

            for prob, next_state, reward, done in env.unwrapped.P[s][action]:

                value += prob * (
                    reward +
                    gamma * V[next_state] * (not done)
                )

            V[s] = value

            delta = max(delta, abs(old - value))

        if delta < theta:
            break

    return V


def policy_improvement(env, V, gamma=0.99):

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    policy = np.zeros(n_states, dtype=int)

    for s in range(n_states):

        q_values = np.zeros(n_actions)

        for a in range(n_actions):

            for prob, next_state, reward, done in env.unwrapped.P[s][a]:

                q_values[a] += prob * (
                    reward +
                    gamma * V[next_state] * (not done)
                )

        policy[s] = np.argmax(q_values)

    return policy


def policy_iteration(env, gamma=0.99):

    n_states = env.observation_space.n

    policy = np.random.randint(
        env.action_space.n,
        size=n_states
    )

    iterations = 0

    while True:

        V = policy_evaluation(
            env,
            policy,
            gamma
        )

        new_policy = policy_improvement(
            env,
            V,
            gamma
        )

        iterations += 1

        if np.array_equal(policy, new_policy):
            break

        policy = new_policy

    return policy, V, iterations


def print_policy(policy):

    arrows = {
        0: "←",
        1: "↓",
        2: "→",
        3: "↑"
    }

    grid = []

    for i in range(16):

        grid.append(arrows[policy[i]])

        if (i + 1) % 4 == 0:
            print(" ".join(grid))
            grid = []


def main():

    env = gym.make(
        "FrozenLake-v1",
        is_slippery=False
    )

    policy, V, steps = policy_iteration(env)

    print("\nConverged in:", steps)

    print("\nOptimal policy:")
    print_policy(policy)

    print("\nState values:")
    print(V.reshape(4, 4))

if __name__ == "__main__":
    main()