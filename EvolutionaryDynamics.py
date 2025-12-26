from math import comb, exp

import numpy as np

from MemoryOneAnalysis import MemoryOneAnalysis
from StochasticGame import StochasticGame
from Strategy import Strategy


class EvolutionaryDynamics:
    def __init__(self, game: StochasticGame, analyzer: MemoryOneAnalysis, group_size, beta):
        self.game = game
        self.analyzer = analyzer
        self.possible_action_combination = []
        self.number_possible_action_combination = len(self.game.possible_strategies) ** self.game.population
        self.group_size = group_size
        self.beta = beta

    def expected_payoffs(self, number_mutants_in_population: int, long_term_payoffs_per_k):
        mutant_payoff = 0.0
        resident_payoff = 0.0
        for k in range(self.group_size):  # loop from resident point of view
            group_probabilities = self.group_probabilities(number_mutants_in_population, k)
            mutant_payoff += group_probabilities[0] * long_term_payoffs_per_k[k + 1][0]
            resident_payoff += group_probabilities[1] * long_term_payoffs_per_k[k][1]
        return mutant_payoff, resident_payoff

    def fixation_probability(self, resident_strategy: Strategy, mutant_strategy: Strategy):
        """

        :param resident_strategy:
        :param mutant_strategy:
        :return:
        """
        alphas = []
        long_term_payoffs_per_k = []
        for k in range(self.group_size + 1):
            long_term_payoffs_per_k.append(
                self.analyzer.get_payoff(self.group_size, k, resident_strategy, mutant_strategy))

        for m in range(1, self.game.population):
            mutant_payoff, resident_payoff = self.expected_payoffs(self.game.population, m, long_term_payoffs_per_k)
            alphas.append(exp(-self.beta * (mutant_payoff - resident_payoff)))
        den = 1.0
        prod = 1.0
        for a in alphas:
            prod *= a
            den += prod

        return 1.0 / den

    def group_probabilities(self, j, k):  # k from resident point of view

        """
        Probability that a focal player sees k mutants in her group

        :param j: Mutants in population
        :param k: Mutant met
        :return:
        """
        numerator_mutant = comb(j, k) * comb(self.game.population - j, self.group_size - k + 1)
        numerator_resident = comb(j , k - 1) * comb(self.game.population - j + 1, self.group_size - k)
        denominator = comb(self.game.population - 1, self.group_size - 1)
        return numerator_mutant / denominator, numerator_resident / denominator
