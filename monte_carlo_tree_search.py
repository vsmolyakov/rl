import math
import random

class TicTacToe:
    def __init__(self):
        self.board = [" "] * 9
        self.current_player = "X"

    def clone(self):
        new_game = TicTacToe()
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        return new_game

    def available_moves(self):
        return [i for i in range(9) if self.board[i] == " "]

    def make_move(self, move):
        self.board[move] = self.current_player
        self.current_player = "O" if self.current_player == "X" else "X"

    def winner(self):
        lines = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8],
            [0, 4, 8],
            [2, 4, 6],
        ]

        for a, b, c in lines:
            if (
                self.board[a] != " "
                and self.board[a] == self.board[b]
                and self.board[a] == self.board[c]
            ):
                return self.board[a]

        if " " not in self.board:
            return "Draw"

        return None

    def terminal(self):
        return self.winner() is not None

    def print_board(self):
        for i in range(0, 9, 3):
            print(self.board[i:i+3])
        print()


class Node:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move

        self.children = []
        self.untried_moves = state.available_moves()

        self.visits = 0
        self.wins = 0

    def fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_child(self, c=1.41):
        scores = []

        for child in self.children:
            exploit = child.wins / child.visits
            explore = math.sqrt(
                math.log(self.visits) / child.visits
            )

            score = exploit + c * explore
            scores.append(score)

        return self.children[scores.index(max(scores))]

    def expand(self):
        move = random.choice(self.untried_moves)
        self.untried_moves.remove(move)

        next_state = self.state.clone()
        next_state.make_move(move)

        child = Node(next_state, parent=self, move=move)
        self.children.append(child)

        return child

    def update(self, result):
        self.visits += 1
        self.wins += result


def mcts(root_state, iterations=500):

    root = Node(root_state)

    root_player = root_state.current_player

    for _ in range(iterations):

        node = root
        state = root_state.clone()

        # Selection
        while (
            not state.terminal()
            and node.fully_expanded()
        ):
            node = node.best_child()
            state.make_move(node.move)

        # Expansion
        if not state.terminal():
            node = node.expand()
            state = node.state.clone()

        # Simulation
        while not state.terminal():
            move = random.choice(state.available_moves())
            state.make_move(move)

        # Backpropagation
        winner = state.winner()

        while node is not None:

            if winner == root_player:
                reward = 1
            elif winner == "Draw":
                reward = 0.5
            else:
                reward = 0

            node.update(reward)
            node = node.parent

    return max(root.children, key=lambda c: c.visits).move


game = TicTacToe()

while not game.terminal():

    if game.current_player == "X":

        move = mcts(game, iterations=1000)

    else:
        move = random.choice(game.available_moves())

    game.make_move(move)
    game.print_board()

print("Winner:", game.winner())