# This code was developed by David Springer in the paper:
# D. Springer et al., "Logistic Regression-HSMM-based Heart Sound
# Segmentation," IEEE Trans. Biomed. Eng., In Press, 2015.
#
# Python implementation.

import numpy as np
from scipy.signal import spectrogram
import matplotlib.pyplot as plt

def get_PSD_feature_Springer_HMM(data, sampling_frequency=1000, frequency_limit_low=40, frequency_limit_high=60, figures=False):
    """
    PSD-based feature extraction for heart sound segmentation.
    
    INPUTS:
    data: this is the audio waveform
    sampling_frequency is self-explanatory
    frequency_limit_low is the lower-bound on the frequency range you want to
    analyse
    frequency_limit_high is the upper-bound on the frequency range
    figures: (optional) boolean variable to display figures
    
    OUTPUTS:
    psd is the array of maximum PSD values between the max and min limits,
    resampled to the same size as the original data.
    (Note: This function returns the raw PSD feature downsampled by the spectrogram
    window step. Resampling to original data length should be done by the caller 
    if needed, as seen in getSpringerPCGFeatures function).
    """
    
    # Ensure data is numpy array
    data = np.array(data)
    
    # Find the spectrogram of the signal:
    # MATLAB: [~,F,T,P] = spectrogram(data,sampling_frequency/40,round(sampling_frequency/80),1:1:round(sampling_frequency/2),sampling_frequency);
    
    # Window length: fs/40 (25ms). 1000/40 = 25.
    nperseg = int(sampling_frequency / 40)
    
    # Overlap: fs/80 (12.5ms). MATLAB rounds 12.5 to 13.
    # Python int(12.5) is 12. Using int(x+0.5) to mimic MATLAB round for positive numbers.
    noverlap = int(sampling_frequency / 80 + 0.5)
    
    # Frequencies: 1:1:round(fs/2). This implies 1 Hz resolution.
    # We use nfft=fs to get 1 Hz bins.
    nfft = int(sampling_frequency)
    
    # Run spectrogram
    # Note: scipy returns onesided spectrum by default for real inputs (0 to fs/2)
    # returns f: Array of sample frequencies.
    # returns t: Array of segment times.
    # returns Sxx: Spectrogram of x. By default, the last axis of Sxx corresponds to the segment times.
    # MATLAB spectrogram does not detrend by default, so we set detrend=False.
    # Also, it appears MATLAB's spectrogram when provided with a specific frequency vector (via Goertzel)
    # does not apply the one-sided scaling (x2) that scipy does by default.
    # To match MATLAB's output values (which are approx half of scipy's one-sided default),
    # we set return_onesided=False. This returns the two-sided spectrum (half power in positive freqs).
    f, t, P = spectrogram(data, fs=sampling_frequency, window='hamming', nperseg=nperseg, noverlap=noverlap, nfft=nfft, detrend=False, return_onesided=False, mode='psd')
    
    # When return_onesided=False, f contains negative frequencies too. 
    # We only care about the positive range which is at the beginning of the array for standard FFT order.
    # We filter f and P to keep only positive frequencies to avoid issues with argmin finding negative aliases.
    # However, since we use abs(f - freq), and positive freqs are first, it should pick the first index (positive).
    # But for safety and clarity:
    mask = f >= 0
    f = f[mask]
    P = P[mask, :]
    
    if figures:
        plt.figure()
        # surf(T,F,10*log(P),'edgecolor','none'); axis tight; view(0,90);
        # Using pcolormesh to emulate surf view(0,90)
        # 10*log(P) -> MATLAB log is ln. But practically dB is log10. 
        # Given the code is 10*log, and MATLAB log is natural log, this is 4.34 * ln(P) dB? 
        # Usually 10*log10 is used. I'll stick to np.log (ln) times 10 to be literal to the code, 
        # but standard spectrograms use log10.
        plt.pcolormesh(t, f, 10 * np.log(P + 1e-10), shading='gouraud')
        plt.ylabel('Hz')
        plt.xlabel('Time (Seconds)')
        plt.title('Spectrogram')
        plt.show()

    # Find the indices for the frequency limits
    # [~, low_limit_position] = min(abs(F - frequency_limit_low));
    low_limit_position = np.argmin(np.abs(f - frequency_limit_low))
    high_limit_position = np.argmin(np.abs(f - frequency_limit_high))

    # Find the mean PSD over the frequency range of interest:
    # psd = mean(P(low_limit_position:high_limit_position,:));
    # MATLAB slice includes end index. Python does not, so add 1.
    if low_limit_position > high_limit_position:
        # Fallback if frequencies are weird
        psd = np.zeros(P.shape[1])
    else:
        psd = np.mean(P[low_limit_position : high_limit_position + 1, :], axis=0)

    if figures:
        # t4  = (1:length(psd))./sampling_frequency;
        # t3  = (1:length(data))./sampling_frequency;
        # Note: This logic assumes psd is not resampled yet, so t4 will be very short 
        # and not align with t3 in time if plotted strictly by index/fs. 
        # MATLAB plot would show psd squeezed to the left.
        
        t4 = np.arange(1, len(psd) + 1) / sampling_frequency
        t3 = np.arange(1, len(data) + 1) / sampling_frequency
        
        plt.figure('PSD Feature')
        
        plt.plot(t3, (data - np.mean(data)) / np.std(data), 'c', label='Data')
        # hold on
        plt.plot(t4, (psd - np.mean(psd)) / np.std(psd), 'k', label='PSD Feature')
        
        plt.legend()
        plt.show()

    return psd
