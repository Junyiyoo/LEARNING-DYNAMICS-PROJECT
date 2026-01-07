import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm
import os

# Import custom Numba core and simulation runner modules
import numba_core as core
import simulation_runner as runner

# ==========================================
# 1. PGG Default Parameter Configuration
# ==========================================
GR_SIZE = 4         # Group size (4-player game)
N_POP = 100         # Population size
N_GEN = 10000       # Evolution generations (PGG converges slowly, suggest running longer)
RUNS_PER_POINT = 24 # Number of runs per data point for averaging

# Default Baseline Values
DEF_R1 = 1.6   # Default High Benefit (Note: In PGG, r must be > 1 or even > n to be significant)
DEF_R2 = 1.2   # Default Low Benefit
DEF_C = 1.0    # Cost
DEF_BETA = 100 # Selection Strength (PGG usually requires stronger selection)
DEF_EPS = 0.001 # Error Rate

# ==========================================
# 2. Single Point Calculation Core
# ==========================================
def run_pgg_point(params):
    """
    Runs PGG simulation for a specific set of parameters.
    """
    r1, r2, c, beta, eps = params
    
    # 1. Pre-computation (done within the process to avoid transferring large objects across processes)
    binom_matrix = core.calc_binom(N_POP, GR_SIZE)
    all_strats = core.get_strategies(GR_SIZE)
    
    # 2. Generate random seeds
    seeds = np.random.randint(0, 1e9, RUNS_PER_POINT)
    
    # 3. Construct partial function
    # Note: This calls runner.run_batch_simulation (your original PGG runner)
    func = partial(runner.run_batch_simulation, 
                   n_gen=N_GEN, gr_size=GR_SIZE, beta=beta, 
                   r1=r1, r2=r2, c=c, 
                   binom_matrix=binom_matrix, all_strats=all_strats, epsilon=eps)
    
    # 4. Execute runs serially (parallelism is handled at the outer sweep level)
    results = [func(s) for s in seeds]
    
    # 5. Calculate steady-state average (using the last 50% of generations)
    cut = int(N_GEN * 0.5)
    ms = np.mean([np.mean(r[0][cut:]) for r in results]) # Stochastic
    m1 = np.mean([np.mean(r[1][cut:]) for r in results]) # Only Game 1
    m2 = np.mean([np.mean(r[2][cut:]) for r in results]) # Only Game 2
    
    return ms, m1, m2

# ==========================================
# 3. Parameter Sweep Controller
# ==========================================
def run_pgg_sweep(param_name, values, ax, label_x):
    print(f"Sweeping PGG Parameter: {param_name}...")
    
    param_list = []
    for val in values:
        # Reset to default values
        p = [DEF_R1, DEF_R2, DEF_C, DEF_BETA, DEF_EPS]
        
        # Modify the specific parameter
        if param_name == 'r1': p[0] = val
        elif param_name == 'eps': p[4] = val
        elif param_name == 'beta': p[3] = val
            
        param_list.append(tuple(p))
    
    # Parallel computation
    # Limit CPU cores to prevent overheating
    cpu_count = os.cpu_count()
    safe_cores = max(1, int(cpu_count * 0.7))
    
    with Pool(processes=safe_cores) as pool:
        results = list(tqdm(pool.imap(run_pgg_point, param_list), total=len(values)))
        
    # Unpack results
    ys = np.array([r[0] for r in results]) # Stochastic
    y1 = np.array([r[1] for r in results]) # Game 1
    y2 = np.array([r[2] for r in results]) # Game 2
    
    # Plotting
    ax.plot(values, ys, 'o-', label='Stochastic Game (PGG)', linewidth=2)
    ax.plot(values, y1, 's--', label='Only Game 1 (High r)', alpha=0.7)
    ax.plot(values, y2, 's--', label='Only Game 2 (Low r)', alpha=0.7)
    
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(label_x, fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Handle log scale for specific parameters
    if param_name in ['eps', 'beta']:
        ax.set_xscale('log')

# ==========================================
# 4. Main Execution Block
# ==========================================
if __name__ == '__main__':
    print("Starting generation of PGG (Public Goods Game) parameter robustness analysis plots...")
    print(f"Group Size: {GR_SIZE}, Population: {N_POP}")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # --- Plot 1: Varying Synergy Factor r1 ---
    # In PGG, cooperation is typically beneficial only if r/n > c. 
    # With n=4, c=1, r must be greater than 4 for cooperation to emerge easily.
    # We scan from 1.0 to 6.0.
    r1_vals = np.linspace(1.0, 6.0, 15)
    run_pgg_sweep('r1', r1_vals, axes[0], r'Synergy factor in State 1 ($r_1$)')
    axes[0].set_ylabel('Cooperation Rate', fontsize=12)
    axes[0].set_title('Robustness to Synergy Factor')
    axes[0].legend()
    # Draw a reference line at r=4 (since Group Size=4)
    axes[0].axvline(GR_SIZE, color='gray', linestyle=':', label='r=n')
    
    # --- Plot 2: Varying Error Rate epsilon ---
    eps_vals = np.logspace(-4, 0, 15)
    run_pgg_sweep('eps', eps_vals, axes[1], r'Error Rate ($\epsilon$)')
    axes[1].set_title('Robustness to Error')

    # --- Plot 3: Varying Selection Strength beta ---
    beta_vals = np.logspace(-2, 2, 15)
    run_pgg_sweep('beta', beta_vals, axes[2], r'Selection Strength ($\beta$)')
    axes[2].set_title('Robustness to Selection')
    
    plt.tight_layout()
    plt.show()