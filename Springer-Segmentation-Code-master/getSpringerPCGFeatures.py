# This code was developed by David Springer in the paper:
# D. Springer et al., "Logistic Regression-HSMM-based Heart Sound
# Segmentation," IEEE Trans. Biomed. Eng., In Press, 2015.
#
# Python implementation.

import numpy as np
from scipy.signal import butter, filtfilt, hilbert, resample_poly
from schmidt_spike_removal import schmidt_spike_removal
from homomorphic_envelope_with_hilbert import homomorphic_envelope_with_hilbert
from get_PSD_feature_Springer_HMM import get_PSD_feature_Springer_HMM
from getDWT import getDWT

def getSpringerPCGFeatures(audio_data, fs, figures=False, include_wavelet=False):
    """
    Get the features used in the Springer segmentation algorithm. These 
    features include:
    - The homomorphic envelope (as performed in Schmidt et al's paper)
    - The Hilbert envelope
    - A wavelet-based feature (optional)
    - A PSD-based feature
    
    Inputs:
    audio_data: array of data from which to extract features
    fs: the sampling frequency of the audio data
    figures: (optional) boolean variable dictating the display of figures
    include_wavelet: (optional) boolean to include wavelet feature
    
    Outputs:
    PCG_Features: array of derived features
    featuresFs: the sampling frequency of the derived features
    """
    
    # Check options
    featuresFs = 50 # Downsampled feature sampling frequency
    
    # Ensure numpy array and float type
    audio_data = np.array(audio_data, dtype=float).flatten()
    
    # 25-400Hz 4th order Butterworth band pass
    # Implementation reference from user notebook
    # Note: MATLAB uses Order 2 LPF + Order 2 HPF. This creates an effective Order 4 system.
    # Python's butter(N, 'bandpass') creates an order 2N system.
    # So butter(4, 'bandpass') creates Order 8. 
    # To match MATLAB results, we must use order=2 (creating Order 4 bandpass).
    lowcut = 25.0
    highcut = 400.0
    order = 2
    b, a = butter(order, [lowcut, highcut], btype='bandpass', fs=fs)
    audio_data = filtfilt(b, a, audio_data)
    
    # Spike removal from the original paper:
    audio_data = schmidt_spike_removal(audio_data, fs)
    
    # Find the homomorphic envelope
    homomorphic_envelope = homomorphic_envelope_with_hilbert(audio_data, fs)
    # Downsample the envelope:
    # MATLAB: resample(homomorphic_envelope, featuresFs, Fs)
    # Python: resample_poly(x, up, down)
    # MATLAB resample uses linear extrapolation padding by default to minimize transients.
    # SciPy resample_poly default is zero padding given padtype='constant'. 
    # use padtype='line' to emulate linear extrapolation.
    downsampled_homomorphic_envelope = resample_poly(homomorphic_envelope, featuresFs, fs, padtype='line')
    # normalise the envelope: (x - mean) / std
    # MATLAB std uses N-1 normalization (ddof=1). Numpy uses N (ddof=0) by default.
    downsampled_homomorphic_envelope = (downsampled_homomorphic_envelope - np.mean(downsampled_homomorphic_envelope)) / np.std(downsampled_homomorphic_envelope, ddof=1)
    
    # Hilbert Envelope
    # MATLAB: Hilbert_Envelope function essentially does abs(hilbert(x)) filtering
    # User notebook: np.abs(hilbert(audio_data))
    hilbert_envelope = np.abs(hilbert(audio_data))
    downsampled_hilbert_envelope = resample_poly(hilbert_envelope, featuresFs, fs, padtype='line')
    # normalise
    downsampled_hilbert_envelope = (downsampled_hilbert_envelope - np.mean(downsampled_hilbert_envelope)) / np.std(downsampled_hilbert_envelope, ddof=1)
    
    # Power spectral density feature:
    # MATLAB: 40-60Hz
    psd = get_PSD_feature_Springer_HMM(audio_data, fs, frequency_limit_low=40, frequency_limit_high=60)
    # psd is returned as a 1D array of length corresponding to spectrogram windows.
    # We need to resample it to match the length of the timestamps of the other features (downsampled_homomorphic_envelope)
    psd = resample_poly(psd, len(downsampled_homomorphic_envelope), len(psd), padtype='line')
    # normalise
    psd = (psd - np.mean(psd)) / np.std(psd, ddof=1)

    
    # Wavelet features:
    if include_wavelet:
        wavelet_level = 3
        wavelet_name = 'rbio3.9'
        
        # Audio needs to be longer than 1 second for getDWT to work:
        if len(audio_data) < fs * 1.025:
            pad_length = int(round(0.025 * fs))
            audio_data = np.concatenate([audio_data, np.zeros(pad_length)])
            
        cD, cA = getDWT(audio_data, wavelet_level, wavelet_name)
        
        # MATLAB: abs(cD(wavelet_level,:));
        # getDWT returns cD as matrix (N_levels x Length)
        # Python: cD[wavelet_level-1, :]
        wavelet_feature = np.abs(cD[wavelet_level-1, :])
        
        # MATLAB: wavelet_feature = wavelet_feature(1:length(homomorphic_envelope));
        # Python: [0 : len]
        wavelet_feature = wavelet_feature[:len(homomorphic_envelope)]
        
        downsampled_wavelet = resample_poly(wavelet_feature, featuresFs, fs, padtype='line')
        downsampled_wavelet = (downsampled_wavelet - np.mean(downsampled_wavelet)) / np.std(downsampled_wavelet, ddof=1)

        
        # Combine features
        PCG_Features = np.column_stack([downsampled_homomorphic_envelope, downsampled_hilbert_envelope, psd, downsampled_wavelet])
    else:
        PCG_Features = np.column_stack([downsampled_homomorphic_envelope, downsampled_hilbert_envelope, psd])
        
    return PCG_Features, featuresFs
