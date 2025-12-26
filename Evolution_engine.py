import numpy as np

def get_payoff_and_coop(p, q, b1, b2, c, eps, mode='stochastic'):
    """
    Calculates long-term payoffs and cooperation rates for resident p vs mutant q.
    Solves an 8-state Markov chain: (Env1, CC, CD, DC, DD) and (Env2, CC, CD, DC, DD).
    """
    # Factor in execution error epsilon
    p_eff = p * (1 - 2 * eps) + eps
    q_eff = q * (1 - 2 * eps) + eps
    
    # 8x8 Transition Matrix M
    M = np.zeros((8, 8))
    opp_idx = {0: 0, 1: 2, 2: 1, 3: 3} # Maps opponent's memory-one indices
    
    for s in range(8):
        env_idx = s // 4
        c1 = p_eff[s]
        # Opponent's choice depends on the current environment and their perspective
        c2 = q_eff[env_idx * 4 + opp_idx[s % 4]]
        
        # Outcome probabilities: CC, CD, DC, DD
        probs = [c1 * c2, c1 * (1 - c2), (1 - c1) * c2, (1 - c1) * (1 - c2)]
        for act, prob in enumerate(probs):
            if mode == 'stochastic':
                # Core Mechanism: Only mutual cooperation (CC) returns to Game 1
                next_env = 0 if act == 0 else 1 
            elif mode == 'only_game1': 
                next_env = 0
            else: 
                next_env = 1
            
            M[s, next_env * 4 + act] = prob

    # Find the stationary distribution (eigenvector of eigenvalue 1)
    vals, vecs = np.linalg.eig(M.T)
    v = vecs[:, np.isclose(vals, 1.0)].real[:, 0]
    v /= v.sum()
    
    # Define payoff vectors for Game 1 and Game 2
    u1 = np.array([b1 - c, -c, b1, 0])
    u2 = np.array([b2 - c, -c, b2, 0])
    
    # Calculate resident payoff
    pi_p = np.dot(v[:4], u1) + np.dot(v[4:], u2)
    
    # Calculate mutant payoff (swap CD/DC payoffs for the mutant's perspective)
    u1_m = np.array([b1 - c, b1, -c, 0])
    u2_m = np.array([b2 - c, b2, -c, 0])
    pi_q = np.dot(v[:4], u1_m) + np.dot(v[4:], u2_m)
    
    # Return resident payoff, mutant payoff, and resident cooperation rate
    return pi_p, pi_q, np.dot(v, p_eff)

def get_fixation_probability(pi_m, pi_r, pi_mm, pi_rm, n_pop, beta):
    """
    Calculates the exact fixation probability in a finite population.
    pi_m: mutant vs resident payoff
    pi_r: resident vs resident payoff
    pi_mm: mutant vs mutant payoff
    pi_rm: resident vs mutant payoff
    """
    diffs = []
    for j in range(1, n_pop):
        # f_m: mutant fitness at frequency j/N
        f_m = ((j - 1) * pi_mm + (n_pop - j) * pi_m) / (n_pop - 1)
        # f_r: resident fitness at frequency (N-j)/N
        f_r = (j * pi_rm + (n_pop - j - 1) * pi_r) / (n_pop - 1)
        diffs.append(f_r - f_m)
    
    # Sum of selection differences for the fixation probability formula
    exponent_sums = np.cumsum(beta * np.array(diffs))
    return 1.0 / (1.0 + np.sum(np.exp(np.clip(exponent_sums, -500, 500))))