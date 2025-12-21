from abc import ABC, abstractmethod
from typing import List

class Strategy(ABC):
    @abstractmethod
    def get_cooperation_probability(self, state: int, prev_actions: List[bool], player_index: int, total_players: int) -> float:
        pass

class AllD(Strategy):
    def get_cooperation_probability(self, state, prev_actions, player_index, total_players):
        return 0.0 # P(s,a) = 0

class AllC(Strategy):
    def get_cooperation_probability(self, state, prev_actions, player_index, total_players):
        return 1.0 # P(s,a) = 1

class Grim(Strategy):
    def get_cooperation_probability(self, state, prev_actions, player_index, total_players):
        num_cooperator = prev_actions.count(True) if prev_actions else 0
        # Trigger strategy: cooperate as long as everybody has cooperated (Table S1)
        if not prev_actions:  # First round (encoded as empty history or special state)
            return 1.0
        else:
            if total_players == num_cooperator:
                return 1.0  # Everyone cooperated
            else:
                return 0.0  # Someone defected -> switch to Defect forever
class pTFT(Strategy):
    def get_cooperation_probability(self, state, prev_actions, player_index, total_players):
        num_cooperator = prev_actions.count(True) if prev_actions else 0
        # Proportional Tit-for-Tat (Table S1)
        # Cooperate with probability k / (n-1), where k is cooperating *co-players*.
        if not prev_actions:
            return 1.0
        else:
            # We must subtract self's action to get co-players count.
            # Assuming we pass 'player_index' to this function:
            self_action = prev_actions[player_index]
            k_others = num_cooperator - (1 if self_action else 0)
            if total_players > 1:
                return k_others / (total_players - 1)
            return 1.0  # Fallback for single player (undefined in paper)

class WSLS(Strategy):
    def get_cooperation_probability(self, state, prev_actions, player_index, total_players):
        num_cooperator = prev_actions.count(True) if prev_actions else 0
        # Win-Stay Lose-Shift (Table S1)
        # Cooperate if everyone used the same action (all C or all D)
        if not prev_actions:
            return 1.0
        else:
            if total_players == num_cooperator:  # All C
                return 1.0
            elif num_cooperator == 0:  # All D (Previous payoff was P or 0, usually considered "Win" in WSLS logic if P > S)
                # Note: Standard WSLS cooperates after mutual D.
                return 1.0
            else:
                return 0.0

class Only1(Strategy):
    def get_cooperation_probability(self, state, prev_actions, player_index, total_players):
        # State-dependent strategy (Table S1)
        if state == 0:  # Assuming 0 is s_1
            return 1.0
        else:
            return 0.0
