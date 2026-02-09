# Python implementation of get_duration_distributions.m

import numpy as np
from default_Springer_HSMM_options import default_Springer_HSMM_options

def get_duration_distributions(heartrate, systolic_time):
    """
    Calculates the duration distributions for each heart cycle state.
    
    Inputs:
    heartrate: average heart rate (BPM)
    systolic_time: systolic time interval (seconds)
    
    Outputs:
    d_distributions: list of tuples (mean, variance) for each state (S1, Systole, S2, Diastole)
    max_S1, min_S1, max_S2, min_S2, max_systole, min_systole, max_diastole, min_diastole
    """
    
    springer_options = default_Springer_HSMM_options()
    
    fs = springer_options.audio_segmentation_Fs
    
    # Calculate means and stds (in samples?)
    # MATLAB: round(0.122 * fs)
    mean_S1 = round(0.122 * fs)
    std_S1 = round(0.022 * fs)
    
    mean_S2 = round(0.094 * fs)
    std_S2 = round(0.022 * fs)
    
    # Systole
    # MATLAB: round(systolic_time * fs) - mean_S1
    mean_systole = round(systolic_time * fs) - mean_S1
    std_systole = (25 / 1000.0) * fs
    
    # Diastole
    # MATLAB: ((60/heartrate) - systolic_time - 0.094) * fs
    mean_diastole = ((60.0 / heartrate) - systolic_time - 0.094) * fs
    std_diastole = 0.07 * mean_diastole + (6 / 1000.0) * fs
    
    # Setup d_distributions
    # MATLAB: d_distributions{1,1} = mean, {1,2} = variance (std^2)
    d_distributions = []
    d_distributions.append((mean_S1, std_S1**2))
    d_distributions.append((mean_systole, std_systole**2))
    d_distributions.append((mean_S2, std_S2**2))
    d_distributions.append((mean_diastole, std_diastole**2))
    
    # Min/Max Systole
    # MATLAB: mean_systole - 3*(std_systole+std_S1)
    # Wait, MATLAB code used std_systole + std_S1 for systole bounds?
    # Line 91: min_systole = mean_systole - 3*(std_systole+std_S1);
    min_systole = mean_systole - 3 * (std_systole + std_S1)
    max_systole = mean_systole + 3 * (std_systole + std_S1)
    
    # Min/Max Diastole
    # Line 94: min_diastole = mean_diastole - 3*std_diastole;
    min_diastole = mean_diastole - 3 * std_diastole
    max_diastole = mean_diastole + 3 * std_diastole
    
    # Min/Max S1
    # Line 98: min_S1 = (mean_S1 - 3*(std_S1));
    min_S1 = mean_S1 - 3 * std_S1
    if min_S1 < (fs / 50.0):
        min_S1 = fs / 50.0
        
    max_S1 = mean_S1 + 3 * std_S1
    
    # Min/Max S2
    # Line 103: min_S2 = (mean_S2 - 3*(std_S2));
    min_S2 = mean_S2 - 3 * std_S2
    if min_S2 < (fs / 50.0):
        min_S2 = fs / 50.0
        
    max_S2 = mean_S2 + 3 * std_S2
    
    return d_distributions, max_S1, min_S1, max_S2, min_S2, max_systole, min_systole, max_diastole, min_diastole
