import numpy as np

def matlab_round(x):
    """
    Behaves like MATLAB's round function: rounds to nearest integer, 
    but for .5 cases, rounds away from zero.
    Since we deal with positive time indices, int(x + 0.5) is sufficient.
    """
    return int(x + 0.5)

def labelPCGStates(envelope, s1_positions, s2_positions, sampling_frequency, figures=False):
    """
    This function assigns the state labels to a PCG record. 
    This is based on ECG markers, derived from the R peak and end-T wave locations.
    
    Inputs:
    envelope: The PCG recording envelope (found in getSchmidtPCGFeatures.m)
    s1_positions: The locations of the R peaks (in samples)
    s2_positions: The locations of the end-T waves (in samples)
    sampling_frequency: The sampling frequency of the PCG recording
    figures (optional): boolean variable dictating the display of figures
    
    Output:
    states: An array of the state label for each sample in the feature
    vector. The total number of states is 4. Therefore, this is an array of
    values between 1 and 4.
    State 1 = S1 sound
    State 2 = systole
    State 3 = S2 sound
    State 4 = diastole
    """
    
    # Ensure inputs are numpy arrays/integers
    envelope = np.array(envelope).flatten()
    # Subtract 1 from positions to convert MATLAB 1-based indices to Python 0-based indices
    # We assume inputs are from MATLAB-generated annotation files.
    s1_positions = np.array(s1_positions, dtype=float).flatten() - 1
    s2_positions = np.array(s2_positions, dtype=float).flatten() - 1
    
    length_signal = len(envelope)
    
    states = np.zeros(length_signal)

    # Timing durations from Schmidt:
    mean_S1 = 0.122 * sampling_frequency
    std_S1 = 0.022 * sampling_frequency
    mean_S2 = 0.092 * sampling_frequency
    std_S2 = 0.022 * sampling_frequency

    # Setting the duration from each R-peak to (R-peak+mean_S1) as the first state:
    for i in range(len(s1_positions)):
        # Set an upper bound, incase the window extends over the length of the signal:
        # MATLAB: upper_bound = round(min(length(states), s1_positions(i) + mean_S1));
        # We use calculate the index (0-based), then add 1 for slicing.
        
        pos = s1_positions[i]
        
        # Calculate expected end index (inclusive in MATLAB logic relative to 0-base, so exclusive for Python slice if we just add duration?)
        # MATLAB: round(pos_mat + mean). 
        # Python: round(pos_py + mean). pos_py = pos_mat - 1.
        # res_py = round(pos_mat - 1 + mean) = res_mat - 1.
        # This gives the 0-based index of the last element.
        # So for slice we need +1.
        
        upper_bound_idx = matlab_round(pos + mean_S1)
        upper_bound_idx = min(length_signal - 1, upper_bound_idx) # Clamp to last index
        
        start_idx = matlab_round(max(0, pos))
        
        # Slice: [start : end+1]
        if upper_bound_idx >= start_idx:
            states[start_idx : upper_bound_idx + 1] = 1

    # Set S2 as state 3 depending on position of end T-wave peak in ECG:
    for i in range(len(s2_positions)):
        
        pos_s2 = s2_positions[i]
        
        # find search window of envelope:
        # T-end +- mean+1sd
        margin = np.floor(mean_S2 + std_S2)
        
        # MATLAB: lower_bound = max([s2 - floor(...), 1])
        # Python: 0-based index
        lower_bound = int(max(0, pos_s2 - margin))
        
        # MATLAB: upper_bound = min(len, ceil(s2 + floor(...)))
        # Python: 0-based index
        upper_bound = int(min(length_signal - 1, np.ceil(pos_s2 + margin)))
        
        # Slice in python excludes stop, so use upper_bound + 1
        window_slice = envelope[lower_bound : upper_bound + 1]
        state_slice = states[lower_bound : upper_bound + 1]
        
        # search_window = envelope(...) .* (states(...)~=1)
        search_window = window_slice * (state_slice != 1)
        
        if len(search_window) == 0:
            continue
            
        # Find the maximum value of the envelope in the search window:
        # [~, S2_index] = max(search_window);
        S2_index_rel = np.argmax(search_window)
        
        # Actual index
        S2_index_abs = lower_bound + S2_index_rel
        S2_index_abs = min(length_signal - 1, S2_index_abs)
        
        # Set the states to state 3, centered on the S2 peak
        # MATLAB: ceil(S2_index +((mean_S2)/2))
        half_width = mean_S2 / 2.0
        
        state3_start = int(max(0, np.ceil(S2_index_abs - half_width)))
        state3_end = int(min(length_signal - 1, np.ceil(S2_index_abs + half_width)))
        
        states[state3_start : state3_end + 1] = 3
            
        # Set the spaces between state 3 and the next R peak as state 4:
        
        # diffs = (s1_positions - s2_positions(i));
        diffs = s1_positions - pos_s2
        
        # Exclude negative
        diffs = np.where(diffs < 0, np.inf, diffs)
        
        if np.all(diffs == np.inf):
            end_pos = length_signal # Go to end
            # In MATLAB, end_pos = length(states). Range is ...:length.
            # In Python, slice ... : length works.
        else:
            # MATLAB: [~, index] = min(diffs); end_pos = s1_positions(index) -1;
            # MATLAB end_pos is the index BEFORE the next S1.
            nearest_idx = np.argmin(diffs)
            next_s1_pos = s1_positions[nearest_idx]
            
            # If next_s1_pos is 0-based index of s1. The sample before it is next_s1_pos - 1.
            # Slice ... : next_s1_pos includes up to next_s1_pos - 1.
            # So calculating end_pos is not strictly needed if we use next_s1_pos as slice stop.
            end_pos = int(next_s1_pos) # Slice exclusive stop
            
        # Start of state 4
        # MATLAB: ceil(S2_index +((mean_S2 +(0*std_S2))/2))
        start_S4 = int(np.ceil(S2_index_abs + half_width))
        
        if end_pos > start_S4:
            states[start_S4 : end_pos] = 4
            
    # Setting the first and last sections of the signal
    
    # first_location_of_definite_state = find(states ~= 0, 1)-1; (MATLAB returns index before first non-zero)
    non_zero_indices = np.where(states != 0)[0]
    
    if len(non_zero_indices) > 0:
        first_idx = non_zero_indices[0] # First non-zero index
        
        # MATLAB: if first_loc > 1 (meaning first_idx > 1 in MATLAB, so index 2 or more).
        # In Python, if first_idx > 0.
        if first_idx > 0:
            found_state = states[first_idx]
            # Fill from 0 to first_idx-1. Slice [:first_idx] does exactly that.
            if found_state == 1:
                states[:first_idx] = 4
            elif found_state == 3:
                states[:first_idx] = 2
        
        # Last step down
        last_idx = non_zero_indices[-1]
        
        # if last_idx < end
        if last_idx < length_signal - 1:
            found_state = states[last_idx]
            # Fill from last_idx + 1 to end
            if found_state == 1:
                states[last_idx + 1 :] = 2
            elif found_state == 3:
                states[last_idx + 1 :] = 4

    # Set everywhere else as state 2:
    states[states == 0] = 2
    
    if figures:
        plt.figure('Envelope and labelled states')
        plt.plot(envelope, label='Envelope')
        plt.plot(states, 'r', label='States')
        plt.legend()
        plt.show()
        
    return states
