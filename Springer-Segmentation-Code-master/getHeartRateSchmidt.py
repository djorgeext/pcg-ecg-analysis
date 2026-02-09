# Python implementation of getHeartRateSchmidt.m

import numpy as np
from scipy.signal import butter, filtfilt, correlate
from schmidt_spike_removal import schmidt_spike_removal
from homomorphic_envelope_with_hilbert import homomorphic_envelope_with_hilbert
import matplotlib.pyplot as plt

def getHeartRateSchmidt(audio_data, fs, figures=False):
    """
    Derive the heart rate and the systolic time interval from a PCG recording.
    
    Inputs:
    audio_data: The raw audio data from the PCG recording
    fs: the sampling frequency of the audio recording
    figures: optional boolean to display figures
    
    Outputs:
    heartRate: the heart rate of the PCG in beats per minute
    systolicTimeInterval: the duration of systole, as derived from the
                          autocorrelation function, in seconds
    """
    
    # Ensure audio_data is 1D
    audio_data = audio_data.flatten()
    
    # 25-400Hz 4th order Butterworth band pass
    # MATLAB: butterworth_low_pass_filter(audio_data,2,400,Fs, false) implies order 2 in call, 
    # which usually results in 4th order after filtfilt (2 forward + 2 backward).
    # Same for high pass.
    
    # Low pass
    nyquist = 0.5 * fs
    Wn_low = 400 / nyquist
    b_low, a_low = butter(2, Wn_low, btype='low')
    audio_data = filtfilt(b_low, a_low, audio_data)
    
    # High pass
    Wn_high = 25 / nyquist
    b_high, a_high = butter(2, Wn_high, btype='high')
    audio_data = filtfilt(b_high, a_high, audio_data)
    
    # Spike removal
    audio_data = schmidt_spike_removal(audio_data, fs)
    
    # Homomorphic envelope
    homomorphic_envelope = homomorphic_envelope_with_hilbert(audio_data, fs)
    
    # Autocorrelation
    y = homomorphic_envelope - np.mean(homomorphic_envelope)
    
    # Calculate autocorrelation using full mode
    c = correlate(y, y, mode='full')
    
    # Normalize (equivalent to 'coeff' in MATLAB)
    # The lag0 peak is at index len(y) - 1
    lag0_index = len(y) - 1
    c_norm = c / c[lag0_index]
    
    # Get positive lags only (lag 1 to end)
    # MATLAB: c(length(homomorphic_envelope)+1:end)
    # In Python, valid indices are 0 to len(c)-1. lag0 is at len(y)-1.
    # Positive lags start at len(y).
    signal_autocorrelation = c_norm[len(y):]
    
    # Heart Rate Calculation
    min_index = int(0.5 * fs)
    max_index = int(2 * fs)
    
    # Careful with indices. MATLAB: min_index:max_index (inclusive)
    # Python slicing: [min_index : max_index+1]
    
    # Ensure indices are within bounds
    if max_index > len(signal_autocorrelation):
        max_index = len(signal_autocorrelation)
        
    search_window = signal_autocorrelation[min_index : max_index]
    
    if len(search_window) == 0:
        # Fallback or error handling
        heartRate = 60 # Default to 60 BPM if detection fails?
        # Or raise error. Sticking to simple logic for now.
        index = 0
    else:
        index = np.argmax(search_window)
    
    # true_index relative to signal_autocorrelation start
    true_index = index + min_index
    
    # heartRate = 60 / time_in_seconds
    # time = true_index / fs
    if true_index != 0:
        heartRate = 60 / (true_index / fs)
    else:
        heartRate = 60 # Prevent div by zero
        
    # Systolic Time Interval Calculation
    max_sys_duration = int(round(((60 / heartRate) * fs) / 2))
    min_sys_duration = int(round(0.2 * fs))
    
    # Bounds check
    if max_sys_duration > len(signal_autocorrelation):
        max_sys_duration = len(signal_autocorrelation)
        
    search_window_sys = signal_autocorrelation[min_sys_duration : max_sys_duration + 1]
    
    if len(search_window_sys) == 0:
        pos = 0
    else:
        pos = np.argmax(search_window_sys)
        
    # Relative to signal_autocorrelation start
    systolic_time_index = min_sys_duration + pos
    systolicTimeInterval = systolic_time_index / fs
    
    if figures:
        plt.figure('Heart rate calculation figure')
        plt.plot(signal_autocorrelation)
        plt.plot(true_index, signal_autocorrelation[true_index], 'ro')
        plt.plot(systolic_time_index, signal_autocorrelation[systolic_time_index], 'mo')
        plt.show()
        
    return heartRate, systolicTimeInterval
