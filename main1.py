from random import random
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

from EvolutionaryDynamics import EvolutionaryDynamics
from MemoryOneAnalysis import MemoryOneAnalysis
from PrisonerDilemma import PrisonerDilemma

n_steps = 5000
n_runs = 50




game = PrisonerDilemma([2.0],1.0, 0.001)
coop_time_series = []
freq = np.zeros(8)
analyzer = MemoryOneAnalysis(game)
ev = EvolutionaryDynamics(game, analyzer,1)

coop_time_series_mean = np.zeros(n_steps)
all_runs_data = []

for run in tqdm(range(n_runs)):
    resident = game.generate_strategy()
    resident.set_allways_defect()
    run_data = []

    for t in range(n_steps):
        mutant = game.generate_strategy()
        rho = ev.fixation_probability(resident, mutant)

        if random() < rho:
            resident = mutant
            resident.player_id = 0
        run_data.append(analyzer.cooperation_rate(resident))

    run_data = np.array(run_data)
    all_runs_data.append(run_data)

    # Aggiorna media online
    coop_time_series_mean += (run_data - coop_time_series_mean) / (run + 1)
analyzer.print_cache_statistics()
# Calcola deviazione standard
all_runs_data = np.array(all_runs_data)
coop_time_series_std = np.std(all_runs_data, axis=0)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(coop_time_series_mean, 'b-', linewidth=2, label='Media')
plt.fill_between(
    range(n_steps),
    coop_time_series_mean - coop_time_series_std,
    coop_time_series_mean + coop_time_series_std,
    alpha=0.2,
    color='blue',
    label='±1 SD'
)
plt.xlabel("Evolutionary time step")
plt.ylabel("Average cooperation")
plt.title(f"Media su {n_runs} run")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()