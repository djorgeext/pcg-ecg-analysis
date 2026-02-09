# Python implementation of viterbiDecodePCG_Springer.m

import numpy as np
import math
from scipy.stats import multivariate_normal
from default_Springer_HSMM_options import default_Springer_HSMM_options
from get_duration_distributions import get_duration_distributions
import sys

def viterbiDecodePCG_Springer(observation_sequence, pi_vector, b_matrix, total_obs_distribution, heartrate, systolic_time, fs, figures=False):
    """
    Viterbi decoding with duration dependence.
    
    Inputs:
    observation_sequence: (T, num_features) array
    pi_vector: initial state probabilities
    b_matrix: observation models (list of sklearn LogisticRegression models)
    total_obs_distribution: [mean, cov] of all observations
    heartrate: BPM
    systolic_time: seconds
    fs: sampling frequency
    figures: boolean
    
    Outputs:
    delta, psi, qt
    """
    
    springer_options = default_Springer_HSMM_options()
    
    # Ensure inputs are numpy arrays
    pi_vector = np.array(pi_vector)
    
    T = len(observation_sequence)
    N = 4 # Number of states
    
    # Max duration of a single state (entire heart cycle)
    max_duration_D = int(round((1 * (60.0 / heartrate)) * fs))
    
    # Initialize Delta and Psi
    # Python size: T + max_duration_D - 1
    total_len = T + max_duration_D - 1
    delta = np.full((total_len, N), -np.inf)
    psi = np.zeros((total_len, N), dtype=int)
    psi_duration = np.zeros((total_len, N), dtype=int)
    
    # Setting up observation probs
    observation_probs = np.zeros((T, N))
    
    # Prepare global observation PDF
    try:
        total_mean = np.array(total_obs_distribution[0]).flatten()
        total_cov = np.array(total_obs_distribution[1])
        
        mvn_total = multivariate_normal(mean=total_mean, cov=total_cov, allow_singular=True)
        po_correction_all = mvn_total.pdf(observation_sequence)
    except Exception as e:
        print(f"Error in multivariate_normal: {e}")
        # Fallback for scalar/1D if needed, assuming diagonal
        po_correction_all = np.ones(T)
        
    # Pre-compute intercept for logistic regression if we have coefficients
    # Add column of ones to observation_sequence for intercept multiplication
    obs_with_intercept = np.hstack([np.ones((T, 1)), observation_sequence])

    for n in range(N):
        # Calculate P(state|obs) using logistic regression
        model_or_coeffs = b_matrix[n]
        
        if hasattr(model_or_coeffs, 'predict_proba'):
            # It's an sklearn model object
            pihat = model_or_coeffs.predict_proba(observation_sequence)
            pihat_target = pihat[:, 1]
        else:
            # It's a coefficient array (from MATLAB logic or manual array)
            # Coefficients [intercept, feat1, feat2, ...]
            # MATLAB mnrfit with binary response produces coefficients for log(P(other)/P(target)).
            # i.e., logit = X * b = log( (1-p)/p )
            # p = 1 / (1 + exp(X * b))
            
            coeffs = np.array(model_or_coeffs).flatten()
            logits = np.dot(obs_with_intercept, coeffs)
            
            # Sigmoid
            # 1 / (1 + exp(logits)) gives P(target) if coeffs are log(P(other)/P(target))
            pihat_target = 1.0 / (1.0 + np.exp(logits))

        # P(o|state) = P(state|obs) * P(obs) / P(state_prior)
        
        # Numerical stability: pi_vector[n] shouldn't be 0
        denom = pi_vector[n] if pi_vector[n] > 0 else 1e-10
        
        observation_probs[:, n] = (pihat_target * po_correction_all) / denom
        
    # Duration probabilities
    d_distributions, max_S1, min_S1, max_S2, min_S2, \
    max_sys, min_sys, max_dia, min_dia = get_duration_distributions(heartrate, systolic_time)
    
    # duration_probs: (N, max_duration_D + 1) to handle 1-based d
    # MATLAB: duration_probs(N, 3*Fs). But accessed up to max_duration_D.
    # We will use max_duration_D sized array (plus 1 for 1-based indexing convenience or just map 0->1)
    # Let's use 1-based indexing mapping: d=1 is index 0.
    
    # Or just allocate large enough. 3*Fs is probably safe.
    duration_probs = np.zeros((N, max_duration_D + 1)) 
    duration_sum = np.zeros(N)
    
    for state_j in range(N):
        mean_d, var_d = d_distributions[state_j]
        std_d = np.sqrt(var_d)
        
        # State mapping: 0:S1, 1:Sys, 2:S2, 3:Dia
        # Constraints
        if state_j == 0: # S1
            min_d, max_d = min_S1, max_S1
        elif state_j == 1: # Sys
            min_d, max_d = min_sys, max_sys
        elif state_j == 2: # S2
            min_d, max_d = min_S2, max_S2
        else: # Dia
            min_d, max_d = min_dia, max_dia
            
        # Generate range of d
        d_vals = np.arange(1, max_duration_D + 1)
        # PDF
        # norm.pdf(x, loc, scale)
        pdf_vals = (1.0 / (np.sqrt(2 * np.pi * var_d))) * np.exp(-0.5 * ((d_vals - mean_d)**2 / var_d))
        
        # Apply constraints
        mask = (d_vals < min_d) | (d_vals > max_d)
        pdf_vals[mask] = 2.2250738585072014e-308 # realmin surrogate
        
        duration_probs[state_j, 1:] = pdf_vals
        duration_sum[state_j] = np.sum(pdf_vals)

    # Viterbi Recursion
    
    # Qt
    qt = np.zeros(total_len)
    
    # Initialisation
    # Python index 0 corresponds to t=1 in MATLAB
    
    # delta[0, :] = log(pi) + log(obs[0, :])
    # Protect log(0)
    with np.errstate(divide='ignore'):
        log_pi = np.log(pi_vector + 1e-300)
        log_obs_0 = np.log(observation_probs[0, :] + 1e-300)
        delta[0, :] = log_pi + log_obs_0
        
    psi[0, :] = -1
    
    # Transition matrix (fixed topology)
    # S1->Sys, Sys->S2, S2->Dia, Dia->S1
    # a_matrix in MATLAB:
    # 0 1 0 0 (1->2)
    # 0 0 1 0 (2->3)
    # 0 0 0 1 (3->4)
    # 1 0 0 0 (4->1)
    
    # In my code 0->1, 1->2, 2->3, 3->0
    # prev_states for state j:
    # j=0 (S1) <- 3 (Dia)
    # j=1 (Sys) <- 0 (S1)
    # j=2 (S2) <- 1 (Sys)
    # j=3 (Dia) <- 2 (S2)
    
    prev_state_map = {0: 3, 1: 0, 2: 1, 3: 2}
    
    # The Loop
    # t goes from 1 to T + max_duration_D - 2 (Python indices)
    # corresponding to MATLAB 2 to T + ... - 1
    
    # Pre-calculate log probs for speed
    log_observation_probs = np.log(observation_probs + 1e-300)
    
    # Cumulative sum of log observation probs to compute prod(obs) quickly
    # sum(log(obs[start:end])) = cumsum[end] - cumsum[start-1]
    obs_cumsum = np.cumsum(log_observation_probs, axis=0)
    # Prepend zeros row for easier indexing
    obs_cumsum = np.vstack([np.zeros((1, N)), obs_cumsum])
    
    for t in range(1, total_len):
        for j in range(N):
            # Previous state that transitions to j
            prev_s = prev_state_map[j]
            
            # Loop over duration d
            # d is 1 to max_duration_D
            # We want to find max over d
            
            # Constraints on window:
            # start_t = t - d (MATLAB: t - d + 1? No, MATLAB indices)
            # MATLAB: start_t = t - d.
            # If t=2 (index 1), d=1 -> start_t = 1 (index 0).
            # If d=2 -> start_t = 0.
            
            # In Python index t:
            # window ends at t (inclusive) -> index t
            # window starts at t - d + 1 (index).
            # length is d.
            
            # MATLAB: t is current step index. start_t = t - d.
            # end_t = t.
            # probs = prod(obs(start_t:end_t)). Length: t - (t-d) + 1 = d+1?
            # Wait, MATLAB: start_t = 2 - 1 = 1. end_t = 2. Range 1:2 is length 2.
            # But d=1 should correspond to length 1?
            # RABINER: d is duration in state j. 
            # If we arrive at state j at time t, and stay for d, we started at t-d+1.
            # MATLAB L223: start_t = t - d. 
            # This looks like it implies duration d, but start is exclusive of previous state?
            # Or is it "look back d steps"?
            # Let's perform exact translation of logic.
            
            # MATLAB:
            # start_t = t - d
            # Clamp start_t to 1.
            # end_t = t. clamp to T.
            # probs = prod(obs(start_t:end_t, j))
            
            # Python equivalent (index t):
            # t_matlab = t + 1
            # start_t_matlab = t_matlab - d
            # start_idx = start_t_matlab - 1 = t - d
            
            # end_idx = t
            
            # Clamping
            # if start_idx < 0: start_idx = 0
            # if start_idx > T - 2: start_idx = T - 2 (MATLAB T-1 constraint)
            # Actually MATLAB T-1 constraint seems specific.
            # Let's just clamp to valid observation indices [0, T-1].
            
            best_delta = -np.inf
            best_d = 0
            best_prev_idx = 0 # Wait, prev state is fixed (prev_s), but we need max_index?
            # In Rabiner, we max over prev state i.
            # Here A is sparse (only 1 transition). So max over i is trival (only prev_s).
            # So max_delta comes from delta(start_t, prev_s).
            # MATLAB line 249: max(delta(start_t, :) + log(a_matrix(:, j)))
            # Since a_matrix has 1 at a specific pos, this selects delta(start_t, prev_s).
            # Wait, MATLAB start_t index into DELTA.
            # Delta is size T+...
            
            
            max_d_iter = max_duration_D
            # Optimization: No need to check d where start_idx < 0 heavily if using clamp?
            # Actually, standard Viterbi constraints d.
            
            for d in range(1, max_d_iter + 1):
                # Calculate indices for delta lookback
                # MATLAB: start_t = t - d (index into delta)
                delta_idx = t - d
                
                # Check valid delta index
                if delta_idx < 0: 
                    # If t < d, we can't look back that far unless we handle initial boundary
                    # MATLAB initializes delta(1, :). 
                    # If delta_idx < 0 (MATLAB < 1), map to 1?
                    # MATLAB Line 227: if start_t < 1: start_t = 1.
                    delta_idx = 0
                
                # Calculate indices for Observations window
                # MATLAB: start_t (clamped T-1), end_t (clamped T).
                # Python obs indices [0, T-1].
                
                obs_start_idx = t - d # Same base logic
                if obs_start_idx < 0: obs_start_idx = 0
                if obs_start_idx > T - 2: obs_start_idx = T - 2 # Logic from MATLAB line 230
                
                obs_end_idx = t
                if obs_end_idx > T - 1: obs_end_idx = T - 1
                
                # Check if window is valid
                if obs_start_idx > obs_end_idx:
                    # Should not happen if logic correct
                    emission_log_prob = -np.inf # or 0?
                else:
                    # Sum logs: cumsum[end+1] - cumsum[start]
                    # indices into cumsum are shifted by 1 (cumsum[0] is 0row)
                    # cumsum row k corresponds to sum up to index k-1
                    # we want sum from obs_start_idx to obs_end_idx (inclusive)
                    # range in cumsum: obs_end_idx+1 to obs_start_idx
                    emission_log_prob = obs_cumsum[obs_end_idx+1, j] - obs_cumsum[obs_start_idx, j]
                
                # Duration prob
                # duration_probs[j, d]
                dur_prob = duration_probs[j, d]
                dur_sum = duration_sum[j]
                if dur_prob <= 0 or dur_sum <= 0:
                    log_dur = -np.inf
                else:
                    log_dur = np.log(dur_prob / dur_sum)
                    
                # Previous Delta
                # prev_delta = delta[delta_idx, prev_s]
                # Log(A) = 0 for valid transition
                val = delta[delta_idx, prev_s] + emission_log_prob + log_dur
                
                if val > best_delta:
                    best_delta = val
                    best_d = d
                
            delta[t, j] = best_delta
            psi[t, j] = prev_s # In MATLAB stored index of state, but here it's fixed 1-to-1
            # Wait, MATLAB: [max_delta, max_index] = max(delta(start_t,:)+log(a_matrix(:,j))');
            # max_index is the STATE index.
            # Since A only allows one, max_index is always prev_s (mapped to 1-based).
            # But MATLAB stores it.
            
            psi_duration[t, j] = best_d
            
    # Termination and Backtracking
    
    # MATLAB: temp_delta = delta(T+1:end, :)
    # Python: delta[T:, :]
    if T < total_len:
        temp_delta = delta[T:, :]
    else:
        temp_delta = delta[T-1:T, :] # Fallback
        
    # Find max in temp_delta
    # unraveled index
    idx_linear = np.argmax(temp_delta)
    # unravel
    pos_local, state_idx = np.unravel_index(idx_linear, temp_delta.shape)
    
    # Real position in delta
    pos = pos_local + T
    
    # 1) State is state_idx (already 0-based)
    state = state_idx
    
    # 2) Preceding state
    offset = pos
    # We need to trace back from offset
    
    # Initial alignment
    # MATLAB: onset = offset - psi_duration(offset, state) + 1
    dur = psi_duration[offset, state]
    onset = offset - dur + 1 # MATLAB + 1 for 1-based indexing logic on indices?
    # Python: if offset is end index (inclusive), and duration is dur.
    # start = end - dur + 1?
    # e.g. end=5, dur=2. samples 4, 5. start=4. 5-2+1=4. Yes.
    
    # Fill qt
    # qt[onset : offset+1] = state
    if offset >= len(qt): offset = len(qt) - 1
    
    # Safe fill
    start_fill = max(0, onset)
    end_fill = min(len(qt), offset + 1)
    qt[start_fill : end_fill] = state
    
    # Update for loop
    # preceding_state = psi(offset, state)
    state = psi[offset, state]
    
    # Loop
    count = 0
    # While onset > 2 (MATLAB) -> Python onset > 1 (index 1 is 2nd sample)
    while onset > 1:
        # 2) offset = onset - 1
        offset = onset - 1
        
        # Preceding state
        # preceding_state is already set to `state` variable from previous block?
        # MATLAB: state = preceding_state (at end of block).
        # Inside loop: preceding_state = psi(offset, state)
        
        preceding_node = psi[offset, state]
        
        # 3) Duration lookup at `offset` for `state`
        dur = psi_duration[offset, state]
        onset = offset - dur + 1
        
        # 4) Fill
        start_fill = max(0, onset)
        end_fill = min(len(qt), offset + 1)
        qt[start_fill : end_fill] = state
        
        # Update state
        state = preceding_node
        
        count += 1
        if count > 5000: # Safety
            break
            
    # Return qt trimmed to T
    qt = qt[:T]
    
    # Convert QT to 1-based states for compatibility if needed?
    # The MATLAB States are 1,2,3,4.
    # My python states are 0,1,2,3.
    # I should return 1-based to match expectations if downstream expects it.
    # Run Springer uses `expand_qt`. `expand_qt` just expands values.
    # Normalizing to 1-based is safer for matching MATLAB output.
    qt = qt + 1
    
    return delta, psi, qt

