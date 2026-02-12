import numpy as np


def _matlab_round_pos(x):
    return int(np.floor(x + 0.5))


def expand_qt(original_qt, old_fs, new_fs, new_length):
    original_qt = np.asarray(original_qt).flatten()
    expanded_qt = np.zeros(int(new_length), dtype=float)

    if original_qt.size == 0:
        return expanded_qt

    # MATLAB: indeces_of_changes = find(diff(original_qt));
    # Convert to MATLAB-like 1-based indices for exact arithmetic below.
    indices_of_changes = np.where(np.diff(original_qt) != 0)[0] + 1
    indices_of_changes = np.concatenate([indices_of_changes, [len(original_qt)]])

    start_index = 0  # MATLAB variable (1-based style with 0 sentinel)

    for end_index in indices_of_changes:
        mid_point = _matlab_round_pos((end_index - start_index) / 2.0) + start_index
        mid_point = min(max(mid_point, 1), len(original_qt))

        value_at_mid_point = original_qt[mid_point - 1]

        expanded_start_index = _matlab_round_pos((start_index / old_fs) * new_fs) + 1
        expanded_end_index = _matlab_round_pos((end_index / old_fs) * new_fs)

        if expanded_end_index > new_length:
            expanded_end_index = int(new_length)

        if expanded_start_index <= expanded_end_index and expanded_start_index <= new_length:
            # MATLAB inclusive indexing: expanded_start_index:expanded_end_index
            expanded_qt[expanded_start_index - 1 : expanded_end_index] = value_at_mid_point

        start_index = end_index

    return expanded_qt
