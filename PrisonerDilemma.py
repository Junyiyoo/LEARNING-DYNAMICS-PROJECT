import numpy as np

from StochasticGame import StochasticGame
from Strategy import StrategyPrisonerDilemma

class PrisonerDilemma(StochasticGame):
    def __init__(self, bs: list[float], c: float, epsilon: float):
        super().__init__(player_number=2, num_states=len(bs),epsilon=epsilon)
        self.payoff_matrices = np.zeros((len(bs), 2, 2))
        # Setup Transition Function Q
        # Example from paper (Fig 2a):
        # State 1 (Index 0) -> High Benefit
        # State 2 (Index 1) -> Low Benefit
        # Transitions: Mutual C (k=2) -> State 1. Anything else (k < 2) -> State 2.

        # We fill self.q for every state s and every number of cooperators k
        if self.num_states == 2:
            for s in range(self.num_states):
                for k in range(self.player_number + 1):
                    # Init probs to 0
                    probs = np.zeros(self.num_states)

                    # Logic: If k=2 (all C), go to State 0 (Game 1)
                    # If k < 2 (someone defects), go to State 1 (Game 2)
                    # Note: This logic assumes the "Stochastic Game" from Fig 2a
                    if k == self.player_number:
                        probs[0] = 1.0  # Prob to go to State 0
                        probs[1] = 0.0
                    else:
                        probs[0] = 0.0
                        probs[1] = 1.0  # Prob to go to State 1

                    self.q[(s, k)] = probs

        # Setup Payoff Matrices (SI Eq. 3)
        # Assuming index 0=Cooperate, 1=Defect based on typical game theory convention
        # But your actions are booleans (True=C).
        for i, b in enumerate(bs):
            # u_CC = b - c
            self.payoff_matrices[i][0, 0] = b - c
            # u_CD = -c
            self.payoff_matrices[i][0, 1] = -c
            # u_DC = b
            self.payoff_matrices[i][1, 0] = b
            # u_DD = 0
            self.payoff_matrices[i][1, 1] = 0

    def payoff_function(self, state: int, actions: list[bool]) -> list[float]:
        # Map True(C) -> 0, False(D) -> 1 to match matrix indices above
        idx0 = 0 if actions[0] else 1
        idx1 = 0 if actions[1] else 1

        payoff_1 = self.payoff_matrices[state][idx0, idx1]
        payoff_2 = self.payoff_matrices[state][idx1, idx0]  # Symmetric
        return [float(payoff_1), float(payoff_2)]

    def generate_strategy(self):
        return StrategyPrisonerDilemma(self.num_states, self.player_number, self.epsilon)