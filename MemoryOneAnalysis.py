from typing import List

import numpy as np

from StochasticGame import StochasticGame
from Strategy import Strategy


class MemoryOneAnalysis:
    def __init__(self, game: StochasticGame):

        self.game = game
        self.possible_action_combination = []
        self.number_possible_action_combination = 2 ** self.game.player_number

        # Internal cache for performance
        self._group_payoff_cache = {}
        self._transition_cache = {}  # Save transition matrix of homogeneous population
        self._probability_first_round_cache = {}
        self.cache_tracker = np.zeros(5)
        # Generate all possible action profiles (e.g., [C,C], [C,D]...)
        # This corresponds to the set 'A' in the paper.
        for j in range(self.number_possible_action_combination):
            # zfill ensures leading zeros for correct length
            possible_action_combination_string = (bin(j)[2:].zfill(self.game.player_number))
            self.possible_action_combination.append(bin_to_bool(possible_action_combination_string))

    def _build_transition_matrix(self, strategies: List[Strategy]) -> np.ndarray:
        """
        Constructs the Matrix M (SI Eq. 7)
        The transition probability is a product of environmental transition (Q)
        and strategy transitions (y_k).
        """
        number_possible_chain_states = self.game.num_states * 2 ** self.game.player_number
        transition_matrix = np.zeros((number_possible_chain_states, number_possible_chain_states))

        for prev_state in range(self.game.num_states):
            for next_state in range(self.game.num_states):
                for i, prev_action in enumerate(self.possible_action_combination):

                    for j, new_action in enumerate(self.possible_action_combination):

                        # Calculate the probability that players choose 'new_action' given 'prev_action'
                        # This corresponds to the product term in SI Eq. 7: Product of y_k
                        y = 1.0
                        for k in range(self.game.player_number):
                            # P_k(s, a)
                            Pk = strategies[k].get_cooperation_probability(next_state, prev_action, k)

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

    def _get_transition_matrix(self, strategies: List[Strategy]) -> np.ndarray:
        if len(set(strategies[k].strategy_ID for k in range(len(
                strategies)))) == 1:  # if we are calculating a transition matrix of a homogeneous population
            key = strategies[0].strategy_ID
            if key not in self._transition_cache:
                self._transition_cache[key] = self._build_transition_matrix(strategies)
            else:
                self.cache_tracker[1] += 1
            return self._transition_cache[key]
        else:
            return self._build_transition_matrix(strategies)

    def _build_probability_first_round(self, strategies: List[Strategy]) -> np.ndarray:
        number_possible_chain_states = self.game.num_states * 2 ** self.game.player_number
        V0 = np.zeros(number_possible_chain_states)

        # Paper assumes "In the first round... P(s1, null) = 1" or similar initialization.
        # Eq 9 says v0 is product of z_k if s=s1, else 0.

        start_state = 0  # "Without loss of generality... players find themselves in state s1" (SI Sec 2.1)

        for i, first_round_action in enumerate(self.possible_action_combination):
            # Logic for product of z_k (Eq 10)
            prob_action_profile = 1.0
            for k in range(self.game.player_number):
                # P(s, empty_set)
                Pk = strategies[k].get_cooperation_probability(start_state, [], k)
                zk = Pk if first_round_action[k] else (1.0 - Pk)
                prob_action_profile *= zk

            # Index for (State 1, Action Profile i)
            idx = start_state * len(self.possible_action_combination) + i
            V0[idx] = prob_action_profile
        return V0

    def _probability_first_round(self, strategies: List[Strategy]) -> np.ndarray:
        """
        Calculates initial vector v0 (SI Eq. 9 & 10)
        Assumes starting in state s_1 (index 0).
        """
        if len(set(strategies[k].strategy_ID for k in range(len(
                strategies)))) == 1:  # if we are calculating a transition matrix of a homogeneous population
            key = strategies[0].strategy_ID
            if key not in self._probability_first_round_cache.keys():
                self._probability_first_round_cache[key] = self._build_probability_first_round(strategies)
            else:
                self.cache_tracker[4] += 1
            return self._probability_first_round_cache[key]
        else:
            V0 = self._build_probability_first_round(strategies)
            return V0

    def _calculate_frequency_vector(self, M: np.ndarray, v0: np.ndarray, delta: float) -> np.ndarray:
        """
        SI Eq. 11: v = (1-delta) * v0 * (I - delta * M)^-1
        and case delta->1: v: (M'-I)v=0
        """

        if delta == 1:
            eigenvalues, eigenvectors = np.linalg.eig(M.T)
            idx = np.argmin(np.abs(eigenvalues - 1.0))
            v = np.real(eigenvectors[:, idx])
            v = v / v.sum()
        else:
            dim = M.shape[0]
            I = np.eye(dim)
            # check use of pseudoinverse
            inv_part = np.linalg.inv(I - delta * M)
            v = (1 - delta) * np.dot(v0, inv_part)

        return v

    def _calculate_expected_payoff(self, v: np.ndarray) -> np.ndarray:
        """
        SI Eq. 12: pi = Sum(v_(s,a) * u(s,a))
        """
        payoff = np.zeros(self.game.player_number)

        for state in range(self.game.num_states):
            for i, action in enumerate(self.possible_action_combination):
                # u(s, a)
                payoff_matrix = self.game.payoff_function(state, action)

                # Retrieve frequency v_(s,a)
                line = state * len(self.possible_action_combination) + i
                freq = v[line]

                # Add weighted payoff
                payoff += freq * np.array(
                    payoff_matrix)  # Ensuring payoff_matrix is np array for element-wise addition
        return payoff

    def _get_payoff(self, group_size, k_mutants, resident_strategy: Strategy, mutant_strategy: Strategy) -> (
            float, float):
        strategies = (
                [mutant_strategy] * k_mutants +
                [resident_strategy] * (group_size - k_mutants)
        )
        transition_matrix = self._get_transition_matrix(strategies)
        V0 = self._probability_first_round(strategies)
        v = self._calculate_frequency_vector(transition_matrix, V0, 1)
        payoff = self._calculate_expected_payoff(v)

        cooperation_rate = None

        if k_mutants == 0:
            payoff_mutant = 0
            payoff_resident = payoff[-1]
            coop = 0.0
            for state in range(self.game.num_states):
                for i, action in enumerate(self.possible_action_combination):
                    idx = state * len(self.possible_action_combination) + i
                    freq = v[idx]
                    coop += freq * sum(action)
            cooperation_rate = coop / self.game.player_number
        elif k_mutants == group_size:
            payoff_mutant = payoff[0]
            payoff_resident = 0
            coop = 0.0
            for state in range(self.game.num_states):
                for i, action in enumerate(self.possible_action_combination):
                    idx = state * len(self.possible_action_combination) + i
                    freq = v[idx]
                    coop += freq * sum(action)
            cooperation_rate = coop / self.game.player_number
        else:
            payoff_mutant = payoff[0]
            payoff_resident = payoff[-1]

        return payoff_resident, payoff_mutant, cooperation_rate

    def get_group_payoff(self, group_size, k_mutants, resident_strategy, mutant_strategy):
        key = (id(resident_strategy), id(mutant_strategy), k_mutants)
        if key not in self._group_payoff_cache:
            self._group_payoff_cache[key] = self._get_payoff(
                group_size, k_mutants, resident_strategy, mutant_strategy
            )
        else:
            self.cache_tracker[0] += 1
        return self._group_payoff_cache[key]

    def cooperation_rate(self, strategy: Strategy, delta: float = 1.0) -> float:
        """
        Returns the long-run average cooperation rate when the whole population
        uses the given strategy.
        Corresponds to cvec in calcPay.m
        """

        # homogeneous population
        strategies = [strategy] * self.game.player_number

        # build Markov chain
        M = self._get_transition_matrix(strategies)
        V0 = self._probability_first_round(strategies)
        v = self._calculate_frequency_vector(M, V0, delta)

        # compute cooperation rate
        coop = 0.0
        for state in range(self.game.num_states):
            for i, action in enumerate(self.possible_action_combination):
                idx = state * len(self.possible_action_combination) + i
                freq = v[idx]
                coop += freq * sum(action)
        return coop / self.game.player_number

    def print_cache_statistics(self):
        """Stampa le statistiche di utilizzo di tutte le cache"""
        cache_names = [
            "group_payoff_cache",
            "transition_cache",
            "expected_payoff_cache",
            "frequency_vector_cache",
            "probability_first_round_cache"
        ]

        print("\n" + "=" * 60)
        print("CACHE STATISTICS")
        print("=" * 60)

        total_hits = sum(self.cache_tracker)
        print(f"Total cache hits: {total_hits}\n")

        for i, (name, hits) in enumerate(zip(cache_names, self.cache_tracker)):
            hits = int(hits)  # Tronca la parte decimale
            print(f"{name:30} | Hits: {hits:8d}")

        print("=" * 60)

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
