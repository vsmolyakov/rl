import numpy as np

class GridWorldMDP:
    def __init__(self):
        # Grid dimensions
        self.rows = 3
        self.cols = 3 

        # Actions
        self.actions = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1)
        }

        # Terminal states
        self.goal = (2, 2)
        self.obstacle = (1, 1)

        # Rewards
        self.goal_reward = 10
        self.step_reward = -0.1
        self.obstacle_penalty = -5

        self.gamma = 0.95

    def get_states(self):
        states = []
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) != self.obstacle:
                    states.append((r, c))
        return states

    def is_terminal(self, state):
        return state == self.goal

    def transition(self, state, action):
        if self.is_terminal(state):
            return state, 0

        dr, dc = self.actions[action]
        nr = state[0] + dr
        nc = state[1] + dc

        # Stay in place if out of bounds
        if nr < 0 or nr >= self.rows:
            nr = state[0]

        if nc < 0 or nc >= self.cols:
            nc = state[1]

        next_state = (nr, nc)

        # Obstacle blocks movement
        if next_state == self.obstacle:
            next_state = state

        reward = self.step_reward

        if next_state == self.goal:
            reward = self.goal_reward

        return next_state, reward


def value_iteration(env, theta=1e-6):

    states = env.get_states()

    V = {s: 0 for s in states}
    policy = {}

    while True:
        delta = 0

        for s in states:

            if env.is_terminal(s):
                continue

            old_v = V[s]

            values = []

            for action in env.actions:

                next_state, reward = env.transition(s, action)

                value = reward + env.gamma * V[next_state]

                values.append(value)

            V[s] = max(values)

            delta = max(delta, abs(old_v - V[s]))

        if delta < theta:
            break

    # Extract policy
    for s in states:

        if env.is_terminal(s):
            policy[s] = "GOAL"
            continue

        best_action = None
        best_value = float("-inf")

        for action in env.actions:

            next_state, reward = env.transition(s, action)

            value = reward + env.gamma * V[next_state]

            if value > best_value:
                best_value = value
                best_action = action

        policy[s] = best_action

    return V, policy


def print_policy(policy):

    arrows = {
        "up": "↑",
        "down": "↓",
        "left": "←",
        "right": "→",
        "GOAL": "G"
    }

    for r in range(3):

        row = []

        for c in range(3):

            s = (r, c)

            if s == (1, 1):
                row.append("X")
            else:
                row.append(arrows[policy[s]])

        print(" ".join(row))


# Run
env = GridWorldMDP()

V, policy = value_iteration(env)

print("State Values:")
for k, v in V.items():
    print(k, round(v, 2))

print("\nOptimal Policy:")
print_policy(policy)