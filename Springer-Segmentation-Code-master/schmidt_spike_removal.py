# This code is derived from the paper:
# S. E. Schmidt et al., "Segmentation of heart sound recordings by a
# duration-dependent hidden Markov model," Physiol. Meas., vol. 31,
# no. 4, pp. 513-29, Apr. 2010.
#
# Developed by David Springer for comparison purposes in the paper:
# D. Springer et al., ?Logistic Regression-HSMM-based Heart Sound
# Segmentation,? IEEE Trans. Biomed. Eng., In Press, 2015.
#
# Python implementation.

import numpy as np

def schmidt_spike_removal(original_signal, fs=1000):
    """
    This function removes the spikes in a signal as done by Schmidt et al in
    the paper:
    Schmidt, S. E., Holst-Hansen, C., Graff, C., Toft, E., & Struijk, J. J.
    (2010). Segmentation of heart sound recordings by a duration-dependent
    hidden Markov model. Physiological Measurement, 31(4), 513-29.
    
    The spike removal process works as follows:
    (1) The recording is divided into 500 ms windows.
    (2) The maximum absolute amplitude (MAA) in each window is found.
    (3) If at least one MAA exceeds three times the median value of the MAA's,
    the following steps were carried out. If not continue to point 4.
      (a) The window with the highest MAA was chosen.
      (b) In the chosen window, the location of the MAA point was identified as the top of the noise spike.
      (c) The beginning of the noise spike was defined as the last zero-crossing point before theMAA point.
      (d) The end of the spike was defined as the first zero-crossing point after the maximum point.
      (e) The defined noise spike was replaced by zeroes.
      (f) Resume at step 2.
    (4) Procedure completed.
    
    Inputs:
    original_signal: The original (1D) audio signal array
    fs: the sampling frequency (Hz)
    
    Outputs:
    despiked_signal: the audio signal with any spikes removed.
    """
    
    # Ensure input is numpy array
    original_signal = np.array(original_signal)
    
    # Find the window size (500 ms)
    # round(fs/2) in MATLAB rounds to nearest integer.
    # In Python, round() rounds to nearest even number for .5 cases in Python 3, 
    # but MATLAB rounds away from zero or to nearest int. 
    # For Hz usually integer so simple round is fine.
    windowsize = int(round(fs / 2))
    
    # Find any samples outside of a integer number of windows:
    trailingsamples = len(original_signal) % windowsize
    
    # Reshape the signal into a number of windows:
    if trailingsamples > 0:
        signal_cut = original_signal[:-trailingsamples]
    else:
        signal_cut = original_signal
        
    # We create a (windowsize, n_windows) array where each column is a window
    # reshape(-1, windowsize) -> (n_windows, windowsize)
    # .T -> (windowsize, n_windows)
    sampleframes = signal_cut.reshape(-1, windowsize).T.copy()
    
    # Find the MAAs:
    MAAs = np.max(np.abs(sampleframes), axis=0)
    
    # While there are still samples greater than 3* the median value of the
    # MAAs, then remove those spikes:
    while np.any(MAAs > np.median(MAAs) * 3):
        
        # Find the window with the max MAA:
        window_num = np.argmax(MAAs)
        
        # Find the postion of the spike within that window:
        current_window = sampleframes[:, window_num]
        spike_position = np.argmax(np.abs(current_window))
        
        # Finding zero crossings (where there may not be actual 0 values, just a change from positive to negative):
        # MATLAB: [abs(diff(sign(sampleframes(:,window_num))))>1; 0]
        diff_sign = np.diff(np.sign(current_window))
        zero_crossings = np.concatenate([np.abs(diff_sign) > 1, [False]])
        
        # Find the start of the spike, finding the last zero crossing before
        # spike position. If that is empty, take the start of the window:
        # Check indices 0 to spike_position inclusive
        candidates_start = np.where(zero_crossings[:spike_position+1])[0]
        
        if len(candidates_start) > 0:
            spike_start = candidates_start[-1]
        else:
            spike_start = 0
            
        # Find the end of the spike, finding the first zero crossing after
        # spike position. If that is empty, take the end of the window:
        
        # Ignore crossings up to spike_position
        # We search in zero_crossings starting from spike_position + 1?
        # No, we search in the whole array but ignore indices <= spike_position.
        # But wait, logic in MATLAB: zero_crossings(1:spike_position) = 0;
        # then find(zero_crossings, 1, 'first').
        
        candidates_after = np.where(zero_crossings[spike_position+1:])[0]
        if len(candidates_after) > 0:
            # candidate index is relative to the slice starting at spike_position+1
            spike_end = candidates_after[0] + (spike_position + 1)
        else:
            spike_end = windowsize - 1
            
        # Set to Zero (using 0.0001)
        sampleframes[spike_start : spike_end + 1, window_num] = 0.0001
        
        # Recalculate MAAs
        MAAs = np.max(np.abs(sampleframes), axis=0)
        
    # Reconstruct signal
    # Flatten column-wise: transpose back to (n_windows, windowsize) then flatten
    despiked_signal = sampleframes.T.flatten()
    
    # Add the trailing samples back to the signal:
    if trailingsamples > 0:
        despiked_signal = np.concatenate([despiked_signal, original_signal[-trailingsamples:]])
        
    return despiked_signal