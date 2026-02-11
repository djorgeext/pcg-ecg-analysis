# This code was developed by David Springer for comparison purposes in the
# paper:
# D. Springer et al., "Logistic Regression-HSMM-based Heart Sound
# Segmentation," IEEE Trans. Biomed. Eng., In Press, 2015.
#
# Python implementation.

import numpy as np
import matplotlib.pyplot as plt
from getSpringerPCGFeatures import getSpringerPCGFeatures
from labelPCGStates import labelPCGStates
from trainBandPiMatricesSpringer import trainBandPiMatricesSpringer

def trainSpringerSegmentationAlgorithm(pcg_audio_list, annotations_list, fs, figures=False):
    """
    Training the Springer HMM segmentation algorithm.
    
    Inputs:
    pcg_audio_list: A list of the N audio signals (numpy arrays). For evaluation
    purposes, these signals should be from a distinct training set of
    recordings, while the algorithm should be evaluated on a separate test
    set of recordings.
    
    annotations_list: a Nx2 list/tuple: item[0] = the positions of the
    R-peaks and item[1] = the positions of the end-T-waves
    (both in SAMPLES).
    
    fs: The sampling frequency of the PCG signals
    figures (optional): boolean variable dictating the display of figures.
    
    Outputs:
    logistic_regression_B_matrix:
    pi_vector:
    total_obs_distribution:
    As Springer et al's algorithm is a duration dependant HMM, there is no
    need to calculate the A_matrix, as the transition between states is only
    dependant on the state durations.
    """
    
    number_of_states = 4
    num_pcgs = len(pcg_audio_list)
    
    # A matrix (list of lists) of the values from each state in each of the PCG recordings:
    # state_observation_values[rec_idx][state_idx]
    state_observation_values = [[None for _ in range(number_of_states)] for _ in range(num_pcgs)]
    
    for i in range(num_pcgs):
        pcg_audio = pcg_audio_list[i]
        
        # Annotations
        s1_locations = annotations_list[i][0]
        s2_locations = annotations_list[i][1]
        
        # Get Features
        pcg_features, features_fs = getSpringerPCGFeatures(pcg_audio, fs, figures=False, include_wavelet=True)
        
        # Label States
        # The first column of PCG_Features is the Homomorphic Envelope
        pcg_states = labelPCGStates(pcg_features[:, 0], s1_locations, s2_locations, features_fs, figures=False)
        
        # Plotting assigned states:
        if figures:
            plt.figure('Assigned states to PCG')
            
            t1 = np.arange(len(pcg_audio)) / fs
            t2 = np.arange(len(pcg_features)) / features_fs
            
            plt.plot(t1, pcg_audio, 'k-', label='Audio')
            plt.plot(t2, pcg_features, label='Features') # Will plot multiple lines
            plt.plot(t2, pcg_states, 'r-', label='States')
            
            plt.legend()
            plt.show()
            
        # Group together all observations from the same state in the PCG recordings:
        for state_i in range(1, number_of_states + 1):
            # MATLAB states are 1, 2, 3, 4
            # Python list index 0, 1, 2, 3
            
            mask = (pcg_states == state_i)
            
            if np.any(mask):
                state_observation_values[i][state_i - 1] = pcg_features[mask, :]
            else:
                 state_observation_values[i][state_i - 1] = np.empty((0, pcg_features.shape[1]))

    # Train the B and pi matrices after all the PCG recordings have been labelled:
    logistic_regression_B_matrix, pi_vector, total_obs_distribution = trainBandPiMatricesSpringer(state_observation_values)
    
    return logistic_regression_B_matrix, pi_vector, total_obs_distribution
