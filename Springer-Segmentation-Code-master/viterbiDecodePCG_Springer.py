import numpy as np
from scipy.stats import multivariate_normal
from get_duration_distributions import get_duration_distributions


def _realmin():
    return np.finfo(float).tiny


def _matlab_round_pos(x):
    return int(np.floor(x + 0.5))


def _compute_p_state_given_obs(observation_sequence, model_or_coeffs):
    if hasattr(model_or_coeffs, "predict_proba"):
        pihat = model_or_coeffs.predict_proba(observation_sequence)
        return pihat[:, 1]

    coeffs = np.array(model_or_coeffs, dtype=float).flatten()
    x = np.hstack([np.ones((observation_sequence.shape[0], 1)), observation_sequence])
    logits = x @ coeffs
    logits = np.clip(logits, -700, 700)
    return 1.0 / (1.0 + np.exp(logits))


def viterbiDecodePCG_Springer(
    observation_sequence,
    pi_vector,
    b_matrix,
    total_obs_distribution,
    heartrate,
    systolic_time,
    fs,
    figures=False,
):
    observation_sequence = np.asarray(observation_sequence, dtype=float)
    pi_vector = np.asarray(pi_vector, dtype=float).flatten()

    T = observation_sequence.shape[0]
    N = 4
    max_duration_D = int(_matlab_round_pos((60.0 / heartrate) * fs))

    total_len = T + max_duration_D - 1
    delta = np.full((total_len, N), -np.inf, dtype=float)
    psi = np.zeros((total_len, N), dtype=int)
    psi_duration = np.zeros((total_len, N), dtype=int)

    observation_probs = np.zeros((T, N), dtype=float)

    total_mean = np.asarray(total_obs_distribution[0], dtype=float).flatten()
    total_cov = np.asarray(total_obs_distribution[1], dtype=float)
    mvn_total = multivariate_normal(mean=total_mean, cov=total_cov, allow_singular=True)
    po_correction_all = mvn_total.pdf(observation_sequence)

    for n in range(N):
        p_state_given_obs = _compute_p_state_given_obs(observation_sequence, b_matrix[n])
        denom = pi_vector[n] if pi_vector[n] > 0 else _realmin()
        observation_probs[:, n] = (p_state_given_obs * po_correction_all) / denom

    (
        d_distributions,
        max_S1,
        min_S1,
        max_S2,
        min_S2,
        max_systole,
        min_systole,
        max_diastole,
        min_diastole,
    ) = get_duration_distributions(heartrate, systolic_time)

    duration_probs = np.zeros((N, int(3 * fs)), dtype=float)
    duration_sum = np.zeros(N, dtype=float)

    for state_j in range(N):
        mean_d, var_d = d_distributions[state_j]
        var_d = float(var_d) if float(var_d) > 0 else _realmin()

        for d in range(1, max_duration_D + 1):
            pdf = (1.0 / np.sqrt(2 * np.pi * var_d)) * np.exp(-0.5 * ((d - mean_d) ** 2 / var_d))

            if state_j == 0:
                if d < min_S1 or d > max_S1:
                    pdf = _realmin()
            elif state_j == 1:
                if d < min_systole or d > max_systole:
                    pdf = _realmin()
            elif state_j == 2:
                if d < min_S2 or d > max_S2:
                    pdf = _realmin()
            else:
                if d < min_diastole or d > max_diastole:
                    pdf = _realmin()

            idx = d - 1
            if idx < duration_probs.shape[1]:
                duration_probs[state_j, idx] = pdf

        duration_sum[state_j] = np.sum(duration_probs[state_j, :])

    with np.errstate(divide="ignore"):
        delta[0, :] = np.log(pi_vector) + np.log(np.maximum(observation_probs[0, :], _realmin()))

    psi[0, :] = -1

    a_matrix = np.array(
        [
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
        ],
        dtype=float,
    )
    log_a = np.where(a_matrix > 0, np.log(a_matrix), -np.inf)

    for t_py in range(1, total_len):
        t_mat = t_py + 1

        for j in range(N):
            for d in range(1, max_duration_D + 1):
                start_t_mat = t_mat - d
                if start_t_mat < 1:
                    start_t_mat = 1
                if start_t_mat > T - 1:
                    start_t_mat = T - 1

                end_t_mat = t_mat
                if end_t_mat > T:
                    end_t_mat = T

                start_idx = start_t_mat - 1
                end_idx = end_t_mat - 1

                candidates = delta[start_idx, :] + log_a[:, j]
                max_index = int(np.argmax(candidates))
                max_delta = candidates[max_index]

                probs = np.prod(observation_probs[start_idx : end_idx + 1, j])
                if probs == 0 or not np.isfinite(probs):
                    probs = _realmin()
                emission_probs = np.log(probs)

                d_idx = d - 1
                if d_idx >= duration_probs.shape[1] or duration_sum[j] <= 0:
                    continue

                dur_prob = duration_probs[j, d_idx]
                if dur_prob <= 0:
                    dur_prob = _realmin()

                delta_temp = max_delta + emission_probs + np.log(dur_prob / duration_sum[j])

                if delta_temp > delta[t_py, j]:
                    delta[t_py, j] = delta_temp
                    psi[t_py, j] = max_index
                    psi_duration[t_py, j] = d

    temp_delta = delta[T:, :]
    idx_flat = int(np.argmax(temp_delta))
    pos_local, _ = np.unravel_index(idx_flat, temp_delta.shape)
    pos = pos_local + T

    state = int(np.argmax(delta[pos, :]))
    qt = np.zeros(total_len, dtype=int)

    offset = pos
    onset = offset - psi_duration[offset, state] + 1
    onset = max(onset, 0)
    qt[onset : offset + 1] = state

    state = psi[offset, state]

    count = 0
    while onset > 1:
        offset = onset - 1
        preceding_state = psi[offset, state]
        onset = offset - psi_duration[offset, state] + 1

        if onset < 1:
            onset = 0

        qt[onset : offset + 1] = state
        state = preceding_state

        count += 1
        if count > 1000:
            break

    qt = qt[:T] + 1

    return delta, psi, qt
