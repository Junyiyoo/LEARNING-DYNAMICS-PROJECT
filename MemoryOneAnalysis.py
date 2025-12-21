from typing import List

import numpy as np

from StochasticGame import StochasticGame
from Strategy import Strategy


class MemoryOneAnalysis:
    def __init__(self, game: StochasticGame):
        self.game = game
        self.possible_action_combination = []
        self.number_possible_action_combination = 2 ** self.game.player_number

        # Generate all possible action profiles (e.g., [C,C], [C,D]...)
        # This corresponds to the set 'A' in the paper.
        for j in range(self.number_possible_action_combination):
            # zfill ensures leading zeros for correct length
            possible_action_combination_sting = (bin(j)[2:].zfill(self.game.player_number))
            self.possible_action_combination.append(bin_to_bool(possible_action_combination_sting))

    def build_transition_matrix(self, strategies: List[Strategy]) -> np.ndarray:
        """
        Constructs the Matrix M (SI Eq. 7)
        The transition probability is a product of environmental transition (Q)
        and strategy transitions (y_k).
        """
        number_possible_chain_states = len(self.game.S) * 2 ** self.game.player_number
        transition_matrix = np.zeros((number_possible_chain_states, number_possible_chain_states))

        for prev_state in self.game.S:
            for next_state in self.game.S:
                for i, prev_action in enumerate(self.possible_action_combination):

                    for j, new_action in enumerate(self.possible_action_combination):

                        # Calculate the probability that players choose 'new_action' given 'prev_action'
                        # This corresponds to the product term in SI Eq. 7: Product of y_k
                        y = 1.0
                        for k in range(self.game.player_number):
                            # P_k(s, a)
                            Pk = strategies[k].get_cooperation_probability(prev_state, prev_action, k,
                                                                           self.game.player_number)

                            # Eq 8: y_k = P_k if action is C, 1-P_k if action is D
                            Yk = Pk if new_action[k] else (1.0 - Pk)
                            y *= Yk

                        # Indices for the matrix
                        row_idx = prev_state * len(self.possible_action_combination) + i
                        col_idx = next_state * len(self.possible_action_combination) + j

                        # SI Eq. 7: M = Q(s' | s, a) * Product(y_k)
                        # self.q[(s, k)] gives the vector of probs to move to [state 0, state 1...]
                        prob_next_env_state = self.game.q[(prev_state, prev_action.count(True))][next_state]

                        transition_matrix[row_idx][col_idx] = prob_next_env_state * y
        return transition_matrix

    def probability_first_round(self, strategies: List[str]) -> np.ndarray:
        """
        Calculates initial vector v0 (SI Eq. 9 & 10)
        Assumes starting in state s_1 (index 0).
        """
        number_possible_chain_states = len(self.game.S) * 2 ** self.game.player_number
        V0 = np.zeros(number_possible_chain_states)

        # Paper assumes "In the first round... P(s1, null) = 1" or similar initialization.
        # Eq 9 says v0 is product of z_k if s=s1, else 0.

        start_state = 0  # "Without loss of generality... players find themselves in state s1" (SI Sec 2.1)

        for i, first_round_action in enumerate(self.possible_action_combination):
            # Logic for product of z_k (Eq 10)
            prob_action_profile = 1.0
            for k in range(self.game.player_number):
                # P(s, empty_set)
                Pk = self.game.memory_one_strategies(start_state, [], strategies[k], player_index=k)
                zk = Pk if first_round_action[k] else (1.0 - Pk)
                prob_action_profile *= zk

            # Index for (State 1, Action Profile i)
            idx = start_state * len(self.possible_action_combination) + i
            V0[idx] = prob_action_profile

        return V0

    def calculate_frequency_vector_discounted(self, M: np.ndarray, v0: np.ndarray, delta: float) -> np.ndarray:
        """
        SI Eq. 11: v = (1-delta) * v0 * (I - delta * M)^-1
        """
        dim = M.shape[0]
        I = np.eye(dim)
        # check use of pseudoinverse
        inv_part = np.linalg.inv(I - delta * M)
        v = (1 - delta) * np.dot(v0, inv_part)
        return v

    def calculate_payoff(self, v: np.ndarray) -> np.ndarray:
        """
        SI Eq. 12: pi = Sum(v_(s,a) * u(s,a))
        """
        payoff = np.zeros(self.game.player_number)

        for state in self.S:
            for i, action in enumerate(self.possible_action_combination):
                # u(s, a)
                payoff_matrix = self.game.payoff_function(state, action)

                # Retrieve frequency v_(s,a)
                line = state * len(self.possible_action_combination) + i
                freq = v[line]

                # Add weighted payoff
                payoff += freq * np.array(payoff_matrix)  # Ensuring payoff_matrix is np array for element-wise addition

        return payoff

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
