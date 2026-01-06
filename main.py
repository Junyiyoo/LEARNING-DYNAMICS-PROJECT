import numpy as np
import matplotlib.pyplot as plt

from functools import partial
from multiprocessing import Pool
from tqdm import tqdm

from Evolution_engine import get_payoff_and_coop, get_fixation_probability
import numba_core as core
import simulation_runner as runner

# --- 1. PD Core Parameters  ---
B1, B2, C = 2.0, 1.2, 1.0  # Benefit and cost
N = 100                    # Population size
BETA = 1.0                 # Selection intensity
EPS = 0.001                # Execution error
STEPS = 5000               # Time steps (number of mutants)
RUNS = 100                 # Number of independent simulations to average

# --- 2. PGG Parameters ---
# Parameters for the new Numba simulation
GR_SIZE = 4       # Group Size
BETA_GROUP = 100.0 # Group games usually require higher selection intensity
R1_GROUP = 1.6
R2_GROUP = 1.2
COST_GROUP = 1.0
N_GEN_GROUP = 10000 # Group games usually run for more generations
RUNS_GROUP = 100    # Number of parallel runs


def run_trajectory(mode):
    """Simulates evolutionary dynamics for a specific game mode."""
    print(f"Simulating mode: {mode}...")
    combined_history = np.zeros(STEPS)
    
    for run in range(RUNS):
        # Start with a population of All-Defectors (Memory-1 strategy: 00000000)
        res_strat = np.zeros(8) 
        curr_coop = 0
        
        for t in range(STEPS):
            combined_history[t] += curr_coop
            # Generate a random Memory-1 mutant strategy
            mut_strat = np.random.randint(0, 2, 8)
            
            # 1. Get resident payoff against itself
            pi_rr, _, _ = get_payoff_and_coop(res_strat, res_strat, B1, B2, C, EPS, mode)
            
            # 2. Get resident vs mutant and mutant vs resident payoffs
            pi_rm, pi_mr, _ = get_payoff_and_coop(res_strat, mut_strat, B1, B2, C, EPS, mode)
            
            # 3. Get mutant payoff against itself
            pi_mm, _, _ = get_payoff_and_coop(mut_strat, mut_strat, B1, B2, C, EPS, mode)
            
            # Fixation: Determine if the mutant takes over the population
            # Args: (mut_vs_res, res_vs_res, mut_vs_mut, res_vs_mut)
            p_fix = get_fixation_probability(pi_mr, pi_rr, pi_mm, pi_rm, N, BETA)
            
            if np.random.rand() < p_fix:
                res_strat = mut_strat
                _, _, curr_coop = get_payoff_and_coop(res_strat, res_strat, B1, B2, C, EPS, mode)
                
    return combined_history / RUNS

def run_trajectory_group():
    """
    Run Group Game Simulation (Numba Accelerated)
    """
    print(f"\n[Group] Starting Numba Simulation (Beta={BETA_GROUP})...")

    # 1. Precompute matrices (Handled by Numba Core)
    binom_matrix = core.calc_binom(100, GR_SIZE) # N=100
    all_strats = core.get_strategies(GR_SIZE)

    # 2. Prepare parallel tasks
    seeds = np.random.randint(0, 1000000, RUNS_GROUP)
    
    # Use partial to fix parameters
    func = partial(runner.run_batch_simulation, 
                   n_gen=N_GEN_GROUP, gr_size=GR_SIZE, beta=BETA_GROUP, 
                   r1=R1_GROUP, r2=R2_GROUP, c=COST_GROUP,
                   binom_matrix=binom_matrix, all_strats=all_strats)

    results = []
    # Start process pool
    with Pool() as pool:
        for res in tqdm(pool.imap_unordered(func, seeds), total=RUNS_GROUP, desc="Group Sim"):
            results.append(res)

    # 3. Calculate averages
    print("Averaging Group results...")
    avg_s = np.mean([r[0] for r in results], axis=0)
    avg_1 = np.mean([r[1] for r in results], axis=0)
    avg_2 = np.mean([r[2] for r in results], axis=0)
    
    return avg_s, avg_1, avg_2

if __name__ == "__main__":
    
    # Reproduce Figure 2a curves
    stochastic = run_trajectory('stochastic')
    game1 = run_trajectory('only_game1')
    game2 = run_trajectory('only_game2')

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(stochastic, label='Stochastic game', linewidth=2.5)
    plt.plot(game1, label='Only game 1', alpha=0.7)
    plt.plot(game2, label='Only game 2', alpha=0.7)
    
    plt.ylim(0, 1.05)
    plt.xlabel('Time (Number of Mutants Introduced)')
    plt.ylabel('Cooperation')
    plt.title('Reproduction of Figure 2a (Stochastic Game Feedback)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
    
    # PGG (Public Goods Game)
    avg_s_gr, avg_1_gr, avg_2_gr = run_trajectory_group()
    plt.figure(figsize=(8, 6)) # Create Figure 2
    x_group = np.arange(N_GEN_GROUP)

    plt.plot(x_group, avg_s_gr, label='Stochastic game', linewidth=2.5)
    plt.plot(x_group, avg_1_gr, label='Only game 1 (High r)', linestyle='--')
    plt.plot(x_group, avg_2_gr, label='Only game 2 (Low r)', linestyle='--')

    plt.ylim(-0.02, 1.02)
    plt.xlim(0, N_GEN_GROUP)
    plt.xlabel('Time (Generations)', fontsize=12)
    plt.ylabel('Cooperation Level', fontsize=12)
    plt.title(f'Figure 2: Group Game (Size={GR_SIZE}, Beta={BETA_GROUP})', fontsize=14)

    # Plot aesthetics
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Annotate key parameters on the plot
    plt.text(N_GEN_GROUP*0.6, 0.5, f"$r_1 = {R1_GROUP}$\n$r_2 = {R2_GROUP}$", fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    plt.legend(frameon=False, fontsize=11)
    plt.tight_layout()
    print("Showing Plot 2.")
    plt.show() # Show the second plot