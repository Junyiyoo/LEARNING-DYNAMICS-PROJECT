import numpy as np
from abc import ABC, abstractmethod


class StochasticGame(ABC):
    @abstractmethod
    def __init__(self):
        self.N = {}  # set of player
        self.S = {}  # set of possible states
        self.A = {(j, s): (True, False) for j in self.N for s in
                  self.S}  # set of actions available to each player at state s
        #  True = Cooperation, D = Defection

        self.q = np.zeros(len(self.N)+1)  # Representation transition function by a vector


    @abstractmethod
    def payoff_function(self, state: int, actions: list[bool], amount: list[float]) -> list[float]:
        pass
