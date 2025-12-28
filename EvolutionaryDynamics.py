from math import comb, exp

from MemoryOneAnalysis import MemoryOneAnalysis
from StochasticGame import StochasticGame
from Strategy import Strategy


class EvolutionaryDynamics:
    def __init__(self, game: StochasticGame, analyzer: MemoryOneAnalysis,beta):
        self.game = game
        self.analyzer = analyzer
        self.possible_action_combination = []
        self.number_possible_action_combination = len(self.game.possible_strategies) ** self.game.population
        self.beta = beta


    def expected_payoffs(self, number_mutants_in_population: int, long_term_payoffs_per_k):
        """

        :param number_mutants_in_population: j
        :param long_term_payoffs_per_k: payoff group
        :return:
        """
        mutant_payoff = 0.0
        resident_payoff = 0.0
        for k in range(self.game.groups_size):  # loop from resident point of view
            group_probability_mutant = hypergeometric(self.game.population -1, number_mutants_in_population -1, self.game.groups_size-1,k)
            group_probability_resident = hypergeometric(self.game.population -1, number_mutants_in_population,self.game.groups_size-1,k)

            mutant_payoff += group_probability_mutant * long_term_payoffs_per_k[k + 1][0]
            resident_payoff += group_probability_resident * long_term_payoffs_per_k[k][1]
        return mutant_payoff, resident_payoff

    def fixation_probability(self, resident_strategy: Strategy, mutant_strategy: Strategy):
        """

        :param resident_strategy:
        :param mutant_strategy:
        :return:
        """
        alphas = []
        long_term_payoffs_per_k = []
        for k in range(self.game.groups_size + 1):
            long_term_payoffs_per_k.append(
                self.analyzer.get_group_payoff(self.game.groups_size, k, resident_strategy, mutant_strategy))

        for m in range(1, self.game.population):
            mutant_payoff, resident_payoff = self.expected_payoffs(m, long_term_payoffs_per_k)
            alphas.append(exp(-self.beta * (mutant_payoff - resident_payoff)))
        den = 1.0
        prod = 1.0
        for a in alphas:
            prod *= a
            den += prod

        return 1.0 / den

def hypergeometric(N_pool, n_success_in_pool, n_draws, k_success):
    if k_success < 0: return 0.0

    denom = comb(N_pool, n_draws)
    if denom == 0: return 0.0
    try:
        num = comb(n_success_in_pool, k_success) * comb(N_pool - n_success_in_pool, n_draws - k_success)
        return num / denom
    except ValueError:
        return 0.0