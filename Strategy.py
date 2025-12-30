import random
from typing import List

import numpy as np
from numpy.matlib import empty


class Strategy:
    """
    Assumes binary actions: True (Cooperation) / False (Defection).
    """
    def __init__(self,states: int, players: int, epsilon : float):
        self.epsilon = epsilon
        self.states = states
        self.players = players
        # self.number_possible_strategies = states * len(self.possible_action) * (len(self.players) - 1)
        self.no_prev_action = random.random()
        self.strategy = []


    def create_strategy(self):
        strategy = np.zeros((self.states, self.players + 1))
        for s in range(self.states):
            for b in range(self.players + 1):
                strategy[s][b] = random.random()
        return strategy

    def get_cooperation_probability(self, state: int, prev_actions: List[int]):
        if empty(prev_actions):
            return self.no_prev_action
        number_of_cooperator = prev_actions.count(1)
        return self._apply_execution_error(self.strategy[state][number_of_cooperator])


    def _apply_execution_error(self, p: float) -> float:
        return (1 - self.epsilon) * p + self.epsilon * (1 - p)

class StrategyPrisonerDilemma(Strategy):
    def __init__(self, states: int, players: int, epsilon: float, player_id: int):
        self.player_id = player_id
        self.opponents = players-1
        super().__init__(states, self.opponents, epsilon)
        defective_strategy = self.create_strategy()
        cooperation_strategy = self.create_strategy()
        self. strategy = [defective_strategy, cooperation_strategy]

    def get_cooperation_probability(self, state: int, prev_actions: List[int]):
        if not prev_actions:
            return self.no_prev_action
        self_action = prev_actions[self.player_id]
        prev_actions = prev_actions.copy()
        prev_actions.remove(self_action)
        opponents_actions = prev_actions
        number_of_cooperator = opponents_actions.count(1)
        return self._apply_execution_error(self.strategy[self_action][state][number_of_cooperator])
class StrategyPublicGoodGame(Strategy):
    def __init__(self, states: int, players: int, epsilon: float):
        super().__init__(states, players, epsilon)
        self.strategy = self.create_strategy()