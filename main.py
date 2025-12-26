import numpy as np
import matplotlib.pyplot as plt
from Evolution_engine import get_payoff_and_coop, get_fixation_probability

# --- 1. Core Parameters (Nature Paper Standards) ---
B1, B2, C = 2.0, 1.2, 1.0  # Benefit and cost
N = 100                    # Population size
BETA = 1.0                 # Selection intensity
EPS = 0.001                # Execution error
STEPS = 5000               # Time steps (number of mutants)
RUNS = 100                 # Number of independent simulations to average

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