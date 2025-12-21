import numpy as np

from StochasticGame import StochasticGame


class PublicGoodGame(StochasticGame):
    def __init__(self, rs: list[float], n_players: int, c: float = 1.0):
        super().__init__(num_players=n_players, num_states=len(rs))

        self.rs = rs
        self.cost_of_cooperation = c

        # Transition Function (Fig 2b logic):
        # If all players cooperate (k=N), go to State 0 (High r).
        # Otherwise, go to State 1 (Low r).
        # (Assuming 2 states for simple example, though class allows more)

        for s in self.S:
            for k in range(self.player_number + 1):
                probs = np.zeros(len(self.S))

                if k == self.player_number:
                    probs[0] = 1.0  # Go to Best State
                else:
                    if len(self.S) > 1:
                        probs[1] = 1.0  # Go to Worse State
                    else:
                        probs[0] = 1.0  # Fallback if only 1 state

                self.q[(s, k)] = probs

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