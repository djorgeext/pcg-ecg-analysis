# This code was developed by David Springer for comparison purposes in the
# paper:
# D. Springer et al., ?Logistic Regression-HSMM-based Heart Sound 
# Segmentation,? IEEE Trans. Biomed. Eng., In Press, 2015.
#
# Python implementation.

import numpy as np
from scipy.signal import butter, filtfilt, hilbert
import matplotlib.pyplot as plt

def homomorphic_envelope_with_hilbert(input_signal, sampling_frequency=1000, lpf_frequency=8, figures=False):
    """
    This function finds the homomorphic envelope of a signal, using the method
    described in the following publications:

    S. E. Schmidt et al., ?Segmentation of heart sound recordings by a 
    duration-dependent hidden Markov model.,? Physiol. Meas., vol. 31, no. 4,
    pp. 513?29, Apr. 2010.

    C. Gupta et al., ?Neural network classification of homomorphic segmented
    heart sounds,? Appl. Soft Comput., vol. 7, no. 1, pp. 286?297, Jan. 2007.

    D. Gill et al., ?Detection and identification of heart sounds using 
    homomorphic envelogram and self-organizing probabilistic model,? in 
    Computers in Cardiology, 2005, pp. 957?960.
    (However, these researchers found the homomorphic envelope of shannon
    energy.)

    In I. Rezek and S. Roberts, ?Envelope Extraction via Complex Homomorphic
    Filtering. Technical Report TR-98-9,? London, 1998, the researchers state
    that the singularity at 0 when using the natural logarithm (resulting in
    values of -inf) can be fixed by using a complex valued signal. They
    motivate the use of the Hilbert transform to find the analytic signal,
    which is a converstion of a real-valued signal to a complex-valued
    signal, which is unaffected by the singularity. 

    A zero-phase low-pass Butterworth filter is used to extract the envelope.
    
    Inputs:
    input_signal: the original signal (1D) signal
    sampling_frequency: the signal's sampling frequency (Hz)
    lpf_frequency: the frequency cut-off of the low-pass filter to be used in
    the envelope extraciton (Default = 8 Hz as in Schmidt's publication).
    figures: (optional) boolean variable dictating the display of a figure of
    both the original signal and the extracted envelope:

    Outputs:
    homomorphic_envelope: The homomorphic envelope of the original
    signal (not normalised).
    """
    
    # Ensure input is numpy array
    input_signal = np.array(input_signal)

    # 8Hz, 1st order, Butterworth LPF
    # MATLAB: [B_low,A_low] = butter(1,2*lpf_frequency/sampling_frequency,'low');
    # Wn = 2 * lpf_frequency / sampling_frequency
    
    low = lpf_frequency
    b_low, a_low = butter(1, low, btype='lowpass', fs=sampling_frequency)
    
    # homomorphic_envelope = exp(filtfilt(B_low,A_low,log(abs(hilbert(input_signal)))));
    analytic_signal = hilbert(input_signal)
    
    # Added eps to log to avoid log(0) just in case, though Hilbert theoretically avoids this on real signals often.
    # The comments mention Hilbert helps avoids singularity, so we stick to log(abs(hilbert))
    log_amplitude = np.log(np.abs(analytic_signal))
    
    homomorphic_envelope = np.exp(filtfilt(b_low, a_low, log_amplitude))
    
    # Remove spurious spikes in first sample:
    # homomorphic_envelope(1) = [homomorphic_envelope(2)];
    if len(homomorphic_envelope) > 1:
        homomorphic_envelope[0] = homomorphic_envelope[1]
    
    if figures:
        plt.figure('Homomorphic Envelope')
        plt.plot(input_signal, label='Original Signal')
        plt.plot(homomorphic_envelope, 'r', label='Homomorphic Envelope')
        plt.legend()
        plt.show()
        
    return homomorphic_envelope
