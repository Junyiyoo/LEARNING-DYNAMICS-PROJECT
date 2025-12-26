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

    def __init__(self, num_players: int, num_states: int, possible_strategies: List[Strategy]):
        self.player_number = num_players
        self.S = list(range(num_states))
        self.possible_strategies = possible_strategies
        # Action set: True=C, False=D (SI Section 2.1, iii)
        self.A = {s: (True, False) for s in self.S}

        # Transition function Q: S x A -> Delta(S) (SI Eq. 1 & 2)
        # Dictionary mapping (state, num_cooperators) -> [prob_s1, prob_s2, ...]

        # We initialize with zeros; subclasses must fill this.
        self.q = {(s, k): np.zeros(num_states)
                  for s in self.S
                  for k in range(self.player_number + 1)}

    @abstractmethod
    def payoff_function(self, state: int, actions: List[bool]) -> List[float]:
        """
        Calculates u(s, a) (SI Eq. 3 & 4)
        """
        raise NotImplementedError("Not implemented yet")