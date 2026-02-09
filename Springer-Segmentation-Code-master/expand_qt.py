# Python implementation of expand_qt.m

import numpy as np

def expand_qt(original_qt, old_fs, new_fs, new_length):
    """
    Function to expand the derived HMM states to a higher sampling frequency.
    Python adaptation of expand_qt.m by David Springer.
    
    Inputs:
    original_qt: the original derived states from the HMM (array-like)
    old_fs: the old sampling frequency of the original_qt
    new_fs: the desired sampling frequency
    new_length: the desired length of the qt signal
    
    Outputs:
    expanded_qt: the expanded qt, to the new FS and length (numpy array)
    """
    
    original_qt = np.array(original_qt).flatten()
    expanded_qt = np.zeros(new_length)
    
    # Find indices where changes occur
    # np.diff(a) gives a[i+1] - a[i]
    # np.where returns indices i
    # So changes contains the LAST index of the current block
    changes = np.where(np.diff(original_qt) != 0)[0]
    
    # Append the last index of the array to handle the final segment
    changes = np.concatenate((changes, [len(original_qt) - 1]))
    
    start_index_old = 0
    
    for end_index_old in changes:
        # Value of the current segment
        # Any index between start and end works
        value = original_qt[end_index_old]
        
        # Calculate expanded indices
        # MATLAB: 
        # start: round((start_index_old_matlab-1)/old_fs * new_fs) + 1  <-- wait, MATLAB start_index was prev_end.
        # Let's stick to the logic derived in thought process.
        
        # start_index_old is 0-based index of start of segment.
        # end_index_old is 0-based index of end of segment.
        
        # In MATLAB code:
        # start_index (init 0, then prev end_index).
        # end_index (from list).
        # exp_start = round(start_index ./ old_fs .* new_fs) + 1;
        # exp_end = round(end_index ./ old_fs .* new_fs);
        
        # Python translation:
        # start_time = start_index_old / old_fs
        # end_time = (end_index_old + 1) / old_fs  <-- (+1 because it covers the full sample duration)
        
        # Actually, let's follow MATLAB exactly.
        # MATLAB start_index starts at 0.
        # MATLAB end_index is 1-based index.
        # So MATLAB segment length is end_index - start_index.
        
        # Python:
        # start_index_old (current segment start, 0-based)
        # end_index_old (current segment end, 0-based)
        # duration in samples = end_index_old - start_index_old + 1
        
        # But MATLAB logic uses specific "time" mapping.
        # MATLAB: start_index maps to round((start_index/old)*new).
        # Python: 
        # In first it: start=0. 
        # exp_start_idx = int(round((start_index_old / old_fs) * new_fs))
        
        # MATLAB: end_index is the 1-based index (e.g. 100).
        # exp_end_idx = int(round(((end_index_old + 1) / old_fs) * new_fs))
        # Note: end_index_old is 0-based, so +1 matches MATLAB's 1-based value.
        
        exp_start_idx = int(round((start_index_old / old_fs) * new_fs))
        exp_end_idx = int(round(((end_index_old + 1) / old_fs) * new_fs))
        
        if exp_end_idx > new_length:
            exp_end_idx = new_length
            
        # Fill segment
        # range [start, end)
        if exp_start_idx < new_length:
            expanded_qt[exp_start_idx : exp_end_idx] = value
            
        # Update start for next iteration
        # MATLAB: start_index = end_index; (so next seg starts at old end)
        # Python: 
        start_index_old = end_index_old + 1
        
    return expanded_qt
