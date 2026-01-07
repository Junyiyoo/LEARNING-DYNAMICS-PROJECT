import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm
import time

from Evolution_engine import get_payoff_and_coop, get_fixation_probability

# ==========================================
# 1. Simulation Core (Supports Pure/Stochastic Strategies)
# ==========================================

def run_simulation_core(params):
    """
    General simulation function.
    Runs a single evolutionary simulation based on the provided parameters.
    """
    # Unpack parameters
    mode = params['mode']
    b1 = params['b1']
    b2 = params['b2']
    c = params['c']
    beta = params['beta']
    eps = params['eps']
    N = params['N']
    steps = params['steps']
    strat_type = params.get('strat_type', 'pure') # 'pure' or 'stochastic'
    
    # Initialize strategy (Resident always starts as AllD - Pure Defection)
    # Using AllD as a baseline facilitates evolution.
    res_strat = np.zeros(8) 
    
    # Define the start of the steady state (last 20% of steps)
    steady_state_start = int(steps * 0.8) 
    total_coop = 0.0
    
    for t in range(steps):
        # --- Key Modification: Support for Stochastic Strategies (Fig b) ---
        if strat_type == 'pure':
            # Pure strategy: Integers 0 or 1
            mut_strat = np.random.randint(0, 2, 8).astype(np.float64)
        else:
            # Stochastic strategy: Floating point numbers between 0.0 and 1.0
            mut_strat = np.random.random(8)
        # ----------------------------------
        
        # Calculate Payoffs
        # 1. Resident vs Resident
        pi_rr, _, _ = get_payoff_and_coop(res_strat, res_strat, b1, b2, c, eps, mode)
        # 2. Resident vs Mutant / Mutant vs Resident
        pi_rm, pi_mr, _ = get_payoff_and_coop(res_strat, mut_strat, b1, b2, c, eps, mode)
        # 3. Mutant vs Mutant
        pi_mm, _, _ = get_payoff_and_coop(mut_strat, mut_strat, b1, b2, c, eps, mode)
        
        # Calculate Fixation Probability
        p_fix = get_fixation_probability(pi_mr, pi_rr, pi_mm, pi_rm, N, beta)
        
        # Evolutionary Step
        if np.random.rand() < p_fix:
            res_strat = mut_strat
            
        # Statistics Collection
        # Accumulate cooperation levels only during the steady state
        if t >= steady_state_start:
            _, _, curr_coop = get_payoff_and_coop(res_strat, res_strat, b1, b2, c, eps, mode)
            total_coop += curr_coop
            
    # Return average cooperation level during steady state
    return total_coop / (steps - steady_state_start)

# ==========================================
# 2. Parameter Sweep Controller
# ==========================================

def run_sweep(param_name, values, ax, title, strat_type='pure'):
    """
    Orchestrates the parameter sweep simulation and plotting.
    """
    print(f"Sweeping {param_name} ({strat_type})...")
    
    # Base parameters (Default values from Fig 2a)
    base_params = {
        'b1': 2.0, 'b2': 1.2, 'c': 1.0, 
        'beta': 1.0, 'eps': 0.001, 
        'N': 100, 'steps': 5000,
        'strat_type': strat_type
    }
    
    modes = ['stochastic', 'only_game1', 'only_game2']
    colors = ["green", "blue", "red"]
    styles = ['o-', 's--', 's--']
    
    # Number of runs per data point
    RUNS_PER_POINT = 100 
    
    for i, mode in enumerate(modes):
        y_means = []
        # Iterate over parameter values with a progress bar
        for val in tqdm(values, desc=f"{mode[:4]}", leave=False):
            tasks = []
            for _ in range(RUNS_PER_POINT):
                p = base_params.copy()
                p['mode'] = mode
                
                # Update the specific parameter being swept
                if param_name == 'b1': p['b1'] = val
                elif param_name == 'eps': p['eps'] = val
                elif param_name == 'beta': p['beta'] = val
                
                tasks.append(p)
            
            # Execute simulations in parallel
            with Pool() as pool:
                res = pool.map(run_simulation_core, tasks)
            y_means.append(np.mean(res))
            
        # Plot the results
        ax.plot(values, y_means, styles[i], color=colors[i], label=mode if param_name=='b1' else "", alpha=0.8)
    
    # Configure plot aesthetics
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

# ==========================================
# 3. Main Execution Block
# ==========================================

if __name__ == '__main__':
    # Create a 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # Flatten axes array for easy indexing (0-3)
    ax = axes.flatten()

    # --- Fig a: Varying Benefit (Pure Strategies) ---
    vals_b1 = np.linspace(1.0, 3.0, 10)
    run_sweep('b1', vals_b1, ax[0], 'a. Pure strategies', strat_type='pure')
    ax[0].set_ylabel('Cooperation rate')
    ax[0].set_xlabel('Benefit in State 1 ($b_1$)')
    ax[0].legend()

    # --- Fig b: Varying Benefit (Stochastic Strategies) ---
    # * Note: strat_type is set to 'stochastic' here *
    run_sweep('b1', vals_b1, ax[1], 'b. Stochastic strategies', strat_type='stochastic')
    ax[1].set_ylabel('Cooperation rate')
    ax[1].set_xlabel('Benefit in State 1 ($b_1$)')

    # --- Fig c: Error Rate (Pure) ---
    vals_eps = np.logspace(-4, 0, 10)
    run_sweep('eps', vals_eps, ax[2], 'c. Error rate $\epsilon$', strat_type='pure')
    ax[2].set_xscale('log') # Logarithmic scale for x-axis
    ax[2].set_xlabel('Error rate $\epsilon$')

    # --- Fig d: Selection Strength (Pure) ---
    vals_beta = np.logspace(-2, 2, 10)
    run_sweep('beta', vals_beta, ax[3], 'd. Selection strength $\\beta$', strat_type='pure')
    ax[3].set_xscale('log') # Logarithmic scale for x-axis
    ax[3].set_xlabel('Selection strength $\\beta$')

    print("\nGeneration Complete. Showing plot...")
    plt.show()