import numpy as np
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
import numba_core as core  # Import the core module

def run_batch_simulation(seed, n_gen, gr_size, beta, r1, r2, c, binom_matrix, all_strats):
    """
    This function orchestrates the simulation. It calls Numba functions from the core module.
    """
    n = gr_size
    ns = all_strats.shape[0]

    # Helper function: Calculate homogeneous population payoffs
    # This prepares data for the Numba core functions
    def get_homo_pay(q_v):
        p_h = np.zeros(ns)
        c_h = np.zeros(ns)
        for i in range(ns):
            # Construct homogeneous strategy matrix
            st_h = np.zeros((n, 2*n), dtype=np.int64)
            for k in range(n): st_h[k] = all_strats[i]
            
            # Call Numba core calculation
            p, co = core.calc_pay_numba(st_h, q_v, r1, r2, c)
            p_h[i] = p[0]
            c_h[i] = co
        return p_h, c_h

    # Define environment vectors (q_vec)
    q_s = np.zeros(gr_size + 1); q_s[0] = 1.0 
    q1 = np.ones(gr_size + 1, dtype=np.float64)
    q2 = np.zeros(gr_size + 1, dtype=np.float64)

    # Precompute
    ph_s, ch_s = get_homo_pay(q_s)
    ph_1, ch_1 = get_homo_pay(q1)
    ph_2, ch_2 = get_homo_pay(q2)

    # Run evolutionary process (Call Numba core)
    c_s = core.evol_proc_numba(q_s, r1, r2, c, beta, n_gen, 100, gr_size, binom_matrix, all_strats, ph_s, ch_s, seed)
    c_1 = core.evol_proc_numba(q1, r1, r1, c, beta, n_gen, 100, gr_size, binom_matrix, all_strats, ph_1, ch_1, seed+1)
    c_2 = core.evol_proc_numba(q2, r2, r2, c, beta, n_gen, 100, gr_size, binom_matrix, all_strats, ph_2, ch_2, seed+2)

    return c_s, c_1, c_2