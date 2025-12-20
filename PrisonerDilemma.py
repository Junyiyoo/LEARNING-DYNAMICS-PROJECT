import numpy as np

from StochasticGame import StochasticGame


class PrisonerDilemma(StochasticGame):
    def __init__(self, bs: list[float], c: float):
        super()
        self.payoff_matrices = np.zeros((len(bs), 2, 2))

        # Definition of transition function by a vector
        self.q[0] = 1 # only if both players cooperate the game stay in State 1
        self.q[1] = 0
        self.q[2] = 0

        for i, b in enumerate(bs):
            self.payoff_matrices[i][0, 0] = b - c
            self.payoff_matrices[i][0, 1] = - c
            self.payoff_matrices[i][1, 0] = b
            self.payoff_matrices[i][1, 1] = 0

    def payoff_function(self, state: int, actions: list[bool], amount=0) -> list[float]:
        reward = float(self.payoff_matrices[state][abs(1 - actions[0]), abs(1 - actions[1])])
        return [reward, reward]
