import numpy as np

from StochasticGame import StochasticGame
from Strategy import StrategyPublicGoodGame


class PublicGoodGame(StochasticGame):
    def __init__(self, rs: list[float], c: float, epsilon: float):
        super().__init__(player_number=8, num_states=len(rs), epsilon= epsilon)

        self.rs = rs
        self.cost_of_cooperation = c

        # Transition Function (Fig 2b logic):
        # If all players cooperate (k=N), go to State 0 (High r).
        # Otherwise, go to State 1 (Low r).
        # (Assuming 2 states for simple example, though class allows more)

        if self.num_states == 2:
            for s in range(self.num_states):
                for k in range(self.player_number + 1):
                    probs = np.zeros(self.num_states)
                    if k == self.player_number:
                        probs[0] = 1.0  # Go to Best State
                    else:
                        if self.num_states > 1:
                            probs[1] = 1.0  # Go to Worse State
                        else:
                            probs[0] = 1.0  # Fallback if only 1 state
                    self.q[(s, k)] = probs
        elif self.num_states == 1:
            for s in range(self.num_states):
                for k in range(self.player_number + 1):
                    self.q[(s, k)] = np.array([1.0])
    def payoff_function(self, state: int, actions: list[bool]) -> list[float]:
        # SI Eq. 4
        cooperating_players = actions.count(True)
        payoff = []

        total_pool = cooperating_players * self.cost_of_cooperation * self.rs[state]
        share = total_pool / self.player_number

        for action in actions:
            if action:  # Cooperator
                reward = share - self.cost_of_cooperation
            else:  # Defector
                reward = share
            payoff.append(reward)

        return payoff
    def generate_strategy(self):
        return StrategyPublicGoodGame(self.num_states, self.player_number, self.epsilon)