from random import random

import numpy as np
from matplotlib import pyplot as plt

from EvolutionaryDynamics import EvolutionaryDynamics
from MemoryOneAnalysis import MemoryOneAnalysis
from PrisonerDilemma import PrisonerDilemma

n_steps = 5000


game = PrisonerDilemma([2.0,.2],1.0)
resident = game.get_random_strategy()
coop_time_series = []
freq = np.zeros(len(game.possible_strategies))
analyzer = MemoryOneAnalysis(game)
ev = EvolutionaryDynamics(game, analyzer,1)


for t in range(n_steps):

    # 1. mutazione
    mutant = game.get_random_strategy()

    # 2. fixation probability
    rho = ev.fixation_probability(resident, mutant)

    # 3. evento stocastico
    if random() < rho:
        resident = mutant

    # 4. misura cooperazione
    coop_time_series.append(analyzer.cooperation_rate(resident))

    # 5. abbondanze
    freq[resident] += 1
plt.plot(coop_time_series)
plt.xlabel("Evolutionary time step")
plt.ylabel("Average cooperation")

