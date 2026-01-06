import numpy as np
from scipy.special import comb
from numba import njit

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
import warnings
from numba import njit

warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
import warnings
from numba import njit

warnings.filterwarnings('ignore')

# ==========================================
# 1. Basic Tools (No Numba)
# ==========================================

def get_strategies(n):
    """Generate all strategies (binary representation)"""
    num_strategies = 2**(2*n)
    indices = np.arange(num_strategies)
    # Generate strategy matrix: (Strategies x Bits)
    strategies = ((indices[:, None] & (1 << np.arange(2*n)[::-1])) > 0).astype(int)
    return strategies

def calc_binom(N, n):
    """
    Precompute Hypergeometric/Binomial distribution probabilities.
    bm[row, col] represents the probability of selecting n-1 neighbors 
    containing 'col' mutants from a population of N-1 individuals 
    that has 'row' total mutants.
    
    Note: Python indexing starts at 0, so adjustments are made for storage.
    """
    # N rows (corresponding to 0..N-1 mutants in the population)
    # n+1 columns (corresponding to 0..n mutants in the sample/neighbors)
    bm = np.zeros((N, n + 1))
    for M in range(0, N): # M: Number of other mutants in the population (0 to N-1)
        for k in range(0, n + 1): # k: Number of mutants in the sampled neighbors
            # Logic: Choose k from M mutants, and choose (n-1)-k from (N-1)-M residents.
            # Total pool to choose from is N-1 individuals, selecting n-1 neighbors.

            # Combination formula: C(M, k) * C(N-1-M, n-1-k) / C(N-1, n-1)
            # Using scipy.special.comb

            total_ways = comb(N - 1, n - 1)
            if total_ways == 0:
                bm[M, k] = 0
                continue

            ways_choose_mutants = comb(M, k)
            ways_choose_residents = comb(N - 1 - M, n - 1 - k)

            bm[M, k] = (ways_choose_mutants * ways_choose_residents) / total_ways
    return bm

# ==========================================
# 2. Numba Acceleration Core
# ==========================================

@njit(fastmath=True)
def solve_stationary_distribution(M):
    """Solve for stationary distribution vM = v"""
    dim = M.shape[0]
    A = M.T - np.eye(dim)
    A[-1] = np.ones(dim)
    b = np.zeros(dim)
    b[-1] = 1.0
    try:
        v = np.linalg.solve(A, b)
    except:
        v = np.ones(dim) / dim
    return v

@njit(fastmath=True)
def calc_pay_numba(strategies, q_vec, r1, r2, c):
    n = strategies.shape[0]
    num_states = 1 << (n+1)

    # Precompute strategy action probabilities (with error epsilon)
    ep = 0.001
    str_prob = (1 - ep) * strategies + ep * (1 - strategies)

    pi_round = np.zeros((num_states, n))
    M = np.zeros((num_states, num_states))

    # Construct Transition Matrix and Payoff Matrix
    for row in range(num_states):
        current_env = (row >> n) & 1
        nr_coop_old = 0
        actions = np.zeros(n, dtype=np.int64)

        # Parse actions
        for i in range(n):
            act = (row >> (n - 1 - i)) & 1
            actions[i] = act
            nr_coop_old += act

        # Calculate Payoff 
        # Note: State 1 is Game 2 (Low), State 0 is Game 1 (High)
        # Based on Nature figure logic: Blue(Game1) -> r1, Orange(Game2) -> r2
        # Here assuming binary 0->Game1, 1->Game2
        mult = r2 if current_env == 1 else r1

        for j in range(n):
            pi_round[row, j] = (nr_coop_old * mult / n) - (actions[j] * c)

        # Calculate next state
        # q_vec stores the probability of "Next is State 0 (Game 1)"
        # Index: 4-nr_coop (i.e., 0C -> idx 4, 4C -> idx 0)
        prob_goto_game1 = q_vec[n - nr_coop_old]

        # Deterministic transition (simplified calculation, original paper is deterministic)
        next_env_target = 0 if prob_goto_game1 > 0.5 else 1

        for col in range(num_states):
            next_env = (col >> n) & 1

            if next_env == next_env_target:
                trpr = 1.0
                for i in range(n):
                    i_coop_old = actions[i]
                    # Key index mapping, must align with get_strategies
                    strat_idx = (2 * n) - nr_coop_old - (n - 1) * i_coop_old - 1

                    p_val = str_prob[i, strat_idx]
                    i_coop_next = (col >> (n - 1 - i)) & 1

                    if i_coop_next == 1:
                        trpr *= p_val
                    else:
                        trpr *= (1.0 - p_val)
                M[row, col] = trpr

    # Solve stationary distribution
    v = solve_stationary_distribution(M)
    # Normalize to prevent precision errors
    v = v / np.sum(v)

    # Calculate expected payoff
    pivec = np.zeros(n)
    # Manual dot product
    for s in range(num_states):
        if v[s] > 1e-12:
            for j in range(n):
                pivec[j] += v[s] * pi_round[s, j]

    # Calculate average cooperation rate
    avg_coop = 0.0
    for s in range(num_states):
        if v[s] > 1e-12:
            c_count = 0
            for k in range(n):
                 if ((s >> (n - 1 - k)) & 1): c_count += 1
            avg_coop += v[s] * c_count
    avg_coop /= n

    return pivec, avg_coop

@njit(fastmath=True)
def calc_rho_numba(s1_idx, s2_idx, pay_h, N, n, q_vec, r1, r2, c, beta, binom_matrix, all_strats):
    """
    Rho Calculation: Fixed index misalignment bug.
    """
    pay = np.zeros((n + 1, 2))
    pay[n, 0] = pay_h[s1_idx]
    pay[0, 1] = pay_h[s2_idx]

    st1 = all_strats[s1_idx]
    st2 = all_strats[s2_idx]

    # Construct mixed group strategies and calculate payoffs
    for n_mut in range(1, n):
        # Construct strategy: First n_mut players are S1, remaining n-n_mut are S2
        group_strats = np.zeros((n, 2*n), dtype=np.int64)
        for k in range(n_mut): group_strats[k] = st1
        for k in range(n_mut, n): group_strats[k] = st2

        pi_vec, _ = calc_pay_numba(group_strats, q_vec, r1, r2, c)
        pay[n_mut, 0] = pi_vec[0]  # S1 (Mutant) payoff
        pay[n_mut, 1] = pi_vec[-1] # S2 (Resident) payoff

    log_alpha = np.zeros(N - 1)

    for j in range(1, N): # j: Total number of mutants in the population (1..N-1)

        pi1 = 0.0
        pi2 = 0.0

        # k: Number of mutants in the neighbors (0..n-1)
        for k in range(n):
            # 1. Calculate Expected Payoff for Mutant (pi1)
            # Context: I am a Mutant.
            # There are j-1 other mutants in the population.
            # I select k neighbors from these j-1 individuals.
            # binom_matrix[j-1, k] is the correct probability here!
            # Total mutants in the group = k + 1 (myself)
            prob_m = binom_matrix[j-1, k]
            val_m = pay[k+1, 0]
            pi1 += prob_m * val_m

            # 2. Calculate Expected Payoff for Resident (pi2)
            # Context: I am a Resident.
            # There are j total mutants in the population.
            # I select k neighbors from these j individuals.
            # binom_matrix[j, k] is the correct probability here.
            # Total mutants in the group = k (all form neighbors)
            prob_r = binom_matrix[j, k]
            val_r = pay[k, 1]
            pi2 += prob_r * val_r

        log_alpha[j-1] = -beta * (pi1 - pi2)

    # Log-Sum-Exp Technique
    S = np.zeros(N-1)
    curr = 0.0
    for i in range(N-1):
        curr += log_alpha[i]
        S[i] = curr

    max_S = -1e9
    for val in S:
        if val > max_S: max_S = val

    if max_S > 700:
        return 0.0

    sum_term = 0.0
    for val in S:
        sum_term += np.exp(val)

    rho = 1.0 / (1.0 + sum_term)
    return rho

@njit(fastmath=True)
def evol_proc_numba(q_vec, r1, r2, c, beta, n_gen, N, gr_size, binom_matrix, all_strats, pay_h, coop_h, seed):
    np.random.seed(seed)
    n = gr_size
    ns = all_strats.shape[0]

    res = 0 # Initial state: All Defectors
    coop_history = np.zeros(n_gen)

    for i in range(n_gen):
        mut = np.random.randint(0, ns)
        rho = calc_rho_numba(mut, res, pay_h, N, n, q_vec, r1, r2, c, beta, binom_matrix, all_strats)

        if np.random.random() < rho:
            res = mut
        coop_history[i] = coop_h[res]

    return coop_history