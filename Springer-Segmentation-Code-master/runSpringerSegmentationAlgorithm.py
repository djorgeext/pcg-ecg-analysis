# Python implementation of runSpringerSegmentationAlgorithm.m

import numpy as np
import matplotlib.pyplot as plt
from getSpringerPCGFeatures import getSpringerPCGFeatures
from getHeartRateSchmidt import getHeartRateSchmidt
from viterbiDecodePCG_Springer import viterbiDecodePCG_Springer
from expand_qt import expand_qt
from default_Springer_HSMM_options import default_Springer_HSMM_options

def runSpringerSegmentationAlgorithm(audio_data, fs, b_matrix, pi_vector, total_observation_distribution, figures=False):
    """
    Run the Springer Segmentation Algorithm on a PCG recording.
    
    Inputs:
    audio_data: Signal array
    fs: Sampling frequency
    b_matrix: Trained observation probability models (list of 4 LogisticRegression objs or equivalent)
    pi_vector: Initial state probabilities (array of length 4)
    total_observation_distribution: [mean, cov] of feature data
    figures: Boolean to show plots
    
    Outputs:
    assigned_states: Array of state labels (1-4) at the original sampling frequency 'fs'
    """
    
    # Options
    springer_options = default_Springer_HSMM_options()
    features_fs = springer_options.audio_segmentation_Fs
    
    # 1. Get Features
    # Note: getSpringerPCGFeatures returns features at features_fs (50Hz) and the fs
    pcg_features, features_fs_extracted = getSpringerPCGFeatures(audio_data, fs)
    
    # Ensure usage of the extracted features_fs if needed, or consistency check
    # features_fs was already defined from options (50Hz).
    
    # 2. Get Heart Rate
    heart_rate, systolic_time_interval = getHeartRateSchmidt(audio_data, fs)
    
    if figures:
        print(f"Heart Rate: {heart_rate} BPM")
        print(f"Systolic Time: {systolic_time_interval} s")
        
    # 3. Viterbi Decoding
    # Note: pi_vector and b_matrix come from training
    delta, psi, qt = viterbiDecodePCG_Springer(
        pcg_features, 
        pi_vector, 
        b_matrix, 
        total_observation_distribution, 
        heart_rate, 
        systolic_time_interval, 
        features_fs, 
        figures=figures
    )
    
    # 4. Expand States to original frame measurement
    # qt is at 50Hz. audio_data is at fs (e.g. 1000Hz or loaded fs)
    # We want assigned_states at fs.
    
    assigned_states = expand_qt(qt, features_fs, fs, len(audio_data))
    
    if figures:
        plt.figure("Derived state sequence")
        t = np.arange(len(audio_data)) / fs
        
        # Normalize audio for plotting
        audio_norm = audio_data / np.max(np.abs(audio_data))
        
        plt.plot(t, audio_norm, 'k', label='Audio data')
        plt.plot(t, assigned_states, 'r--', label='Derived states')
        plt.xlabel('Time (s)')
        plt.legend()
        plt.show()
        
    return assigned_states
