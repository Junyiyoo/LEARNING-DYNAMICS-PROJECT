import random

import numpy as np
from abc import abstractmethod
# from parso.python.tree import String # REMOVE: unnecessary dependency
# Use standard typing instead
from typing import List, Dict

from Strategy import Strategy


class StochasticGame:
    """
    Framework for Stochastic Games (Hilbe et al., 2018).
    Assumes binary actions: True (Cooperation) / False (Defection).
    """

    def __init__(self, population: int, num_states: int, possible_strategies: List[Strategy], groups_size = 4):
        self.population = population
        self.S = list(range(num_states))
        self.possible_strategies = possible_strategies
        # Action set: True=C, False=D (SI Section 2.1, iii)
        self.A = {s: (True, False) for s in self.S}
        self.groups_size = groups_size
        # Transition function Q: S x A -> Delta(S) (SI Eq. 1 & 2)
        # Dictionary mapping (state, num_cooperators) -> [prob_s1, prob_s2, ...]

        # We initialize with zeros; subclasses must fill this.
        self.q = {(s, k): np.zeros(num_states)
                  for s in self.S
                  for k in range(self.population + 1)}

    @abstractmethod
    def payoff_function(self, state: int, actions: List[bool]) -> List[float]:
        """
        Calculates u(s, a) (SI Eq. 3 & 4)
        """
        raise NotImplementedError("Not implemented yet")

    def get_random_strategy(self):
        return random.choice(self.possible_strategies)