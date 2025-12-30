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

    def __init__(self, player_number : int, num_states: int, epsilon: float):
        self.player_number = player_number
        self.epsilon = epsilon
        self.num_states = num_states
        # Action set: True=C, False=D (SI Section 2.1, iii)
        self.A = {s: (True, False) for s in range(self.num_states)}
        # Transition function Q: S x A -> Delta(S) (SI Eq. 1 & 2)
        # Dictionary mapping (state, num_cooperators) -> [prob_s1, prob_s2, ...]

        # We initialize with zeros; subclasses must fill this.
        self.q = {(s, k): np.zeros(num_states)
                  for s in range(self.num_states)
                  for k in range(self.player_number + 1)}


    @abstractmethod
    def payoff_function(self, state: int, actions: List[bool]) -> List[float]:
        """
        Calculates u(s, a) (SI Eq. 3 & 4)
        """
        raise NotImplementedError("Not implemented yet")



def bin_to_bool(mu_bin: str):
    """
   Convert a binary string into a list of bool.

   Each '0' is mapped to False, and each '1' is mapped to True.
   """
    bools = []
    for bit in mu_bin:
        if bit == '0':
            bools.append(False)
        if bit == '1':
            bools.append(True)
    return bools