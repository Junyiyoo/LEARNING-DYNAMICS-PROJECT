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
        self.no_prev_action = 1.0 if random.random() > 0.5 else 0.0
        self.strategy = []
        self.strategy_ID = ""

    def _create_strategy(self):
        strategy = np.zeros((self.states, self.players + 1))
        for s in range(self.states):
            for b in range(self.players + 1):
                strategy[s][b] = strategy[s][b] = 1.0 if random.random() > 0.5 else 0.0
        return strategy

    def allways_defect(self):
        strategy = np.zeros((self.states, self.players + 1))
        return strategy

    def get_cooperation_probability(self, state: int, prev_actions: List[int], self_player):
        if empty(prev_actions):
            return self.no_prev_action
        number_of_cooperator = prev_actions.count(1)
        return self._apply_execution_error(self.strategy[state][number_of_cooperator])


    def _apply_execution_error(self, p: float) -> float:
        return (1 - self.epsilon) * p + self.epsilon * (1 - p)

class StrategyPrisonerDilemma(Strategy):
    def __init__(self, states: int, players: int, epsilon: float):
        self.opponents = players-1
        super().__init__(states, self.opponents, epsilon)
        defective_strategy = self._create_strategy()
        cooperation_strategy = self._create_strategy()
        self. strategy = [defective_strategy, cooperation_strategy]
        self._set_strategy_ID()

    def set_allways_defect(self):
        defective_strategy = np.zeros((self.states, self.players + 1))
        cooperation_strategy = np.zeros((self.states, self.players + 1))
        self.strategy = [defective_strategy, cooperation_strategy]


    def get_cooperation_probability(self, state: int, prev_actions: List[int], self_player):
        if not prev_actions:
            return self.no_prev_action
        self_action = prev_actions[self_player]
        prev_actions = prev_actions.copy()
        prev_actions.remove(self_action)
        opponents_actions = prev_actions
        number_of_cooperator = opponents_actions.count(1)
        return self._apply_execution_error(self.strategy[self_action][state][number_of_cooperator])



    def _set_strategy_ID(self):
        self.strategy_ID = ""
        for s in range(self.states):
            for b in range(self.players + 1):
                self.strategy_ID += str(self.strategy[0][s][b])
        for s in range(self.states):
            for b in range(self.players + 1):
                self.strategy_ID += str(self.strategy[1][s][b])


class StrategyPublicGoodGame(Strategy):
    def __init__(self, states: int, players: int, epsilon: float):
        super().__init__(states, players, epsilon)
        self.strategy = self._create_strategy()