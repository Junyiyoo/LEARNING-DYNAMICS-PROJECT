import numpy as np

from StochasticGame import StochasticGame


class PublicGoodGame(StochasticGame):
    def __init__(self, rs: list[float], n_players: int, c: float = 1.0):
        super().__init__(population=n_players, num_states=len(rs))

        self.rs = rs
        self.cost_of_cooperation = c

        # Transition Function (Fig 2b logic):
        # If all players cooperate (k=N), go to State 0 (High r).
        # Otherwise, go to State 1 (Low r).
        # (Assuming 2 states for simple example, though class allows more)

        for s in self.num_states:
            for k in range(self.population + 1):
                probs = np.zeros(len(self.num_states))

                if k == self.population:
                    probs[0] = 1.0  # Go to Best State
                else:
                    if len(self.num_states) > 1:
                        probs[1] = 1.0  # Go to Worse State
                    else:
                        probs[0] = 1.0  # Fallback if only 1 state

                self.q[(s, k)] = probs

    def payoff_function(self, state: int, actions: list[bool]) -> list[float]:
        # SI Eq. 4
        cooperating_players = actions.count(True)
        payoff = []

        total_pool = cooperating_players * self.cost_of_cooperation * self.rs[state]
        share = total_pool / self.population

        for action in actions:
            if action:  # Cooperator
                reward = share - self.cost_of_cooperation
            else:  # Defector
                reward = share
            payoff.append(reward)

        return payoff