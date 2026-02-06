# This code was developed by David Springer for comparison purposes in the
# paper:
# D. Springer et al., "Logistic Regression-HSMM-based Heart Sound
# Segmentation," IEEE Trans. Biomed. Eng., In Press, 2015.
#
# Python implementation.

import numpy as np
import pywt

def wkeep1(x, len_target):
    """
    Mimics MATLAB's wkeep1 function to extract central part of vector.
    """
    n = len(x)
    if n == len_target:
        return x
    elif n > len_target:
        # Extract center
        # Python 0-based indexing:
        # start index is floor((n - len_target) / 2)
        start = (n - len_target) // 2
        return x[start : start + len_target]
    else:
        # If x is shorter, symmetric padding
        diff = len_target - n
        pad_l = diff // 2
        pad_r = diff - pad_l
        return np.pad(x, (pad_l, pad_r), 'constant')

def getDWT(X, N, Name):
    """
    finds the discrete wavelet transform at level N for signal X using the
    wavelet specified by Name.

    Inputs:
    X: the original signal
    N: the decomposition level
    Name: the wavelet name to use

    Outputs:
    cD is a N-row matrix containing the detail coefficients up to N levels
    cA is the same for the approximations
    """
    X = np.array(X)
    len_X = len(X)
    
    # MATLAB's Morlet wavelet handling
    if Name == 'morl':
        # c = cwt(X,1:N,'morl');
        # scales 1 to N
        scales = np.arange(1, N + 1)
        # pywt.cwt syntax: cwt(data, scales, wavelet)
        # returns (coefs, frequencies)
        # Note: MATLAB's complex morlet might be 'cmor' in pywt, but 'morl' is also available.
        # Assuming 'morl' is real Morlet in this context or matching string.
        # MATLAB 'morl' is Real Morlet.
        coefs, _ = pywt.cwt(X, scales, 'morl')
        
        # cD = c; cA = c;
        cD = coefs
        cA = coefs
        return cD, cA
        
    else:
        # Preform wavelet decomposition
        # [c,l] = wavedec(X,N,Name);
        # Python returns list [cA_N, cD_N, ..., cD_1]
        # We need to compute full decomposition up to level N
        coeffs_list = pywt.wavedec(X, Name, level=N)
        # coeffs_list locations: 
        # index 0: cA_N
        # index 1: cD_N
        # ...
        # index N: cD_1
        
        # cD = zeros(N,len);
        cD = np.zeros((N, len_X))
        
        # Reorder the details based on the structure of the wavelet decomposition
        for k in range(1, N + 1):
            # d = detcoef(c,l,k);
            # In MATLAB, k=1 is finest detail (cD1).
            # In Python list, cD1 is at index -1 (or N). cDk is at index -k.
            # Example: N=3. list=[cA3, cD3, cD2, cD1].
            # k=1 (cD1) -> list[-1] (cD1)
            # k=2 (cD2) -> list[-2] (cD2)
            # k=3 (cD3) -> list[-3] (cD3)
            d = coeffs_list[-k]
            
            # d = d(ones(1,2^k),:); 
            # Upsample by repeating each element 2^k times
            d_up = np.repeat(d, 2**k)
            
            # cD(k,:) = wkeep1(d(:)',len);
            cD[k-1, :] = wkeep1(d_up, len_X)
            
        # Space cD according to spacing of floating point numbers:
        # I = find(abs(cD)<sqrt(eps)); cD(I) = zeros(size(I));
        eps = np.finfo(float).eps
        threshold = np.sqrt(eps)
        cD[np.abs(cD) < threshold] = 0
        
        # Reorder the approximations 
        # cA = zeros(N,len);
        cA = np.zeros((N, len_X))
        
        for k in range(1, N + 1):
            # a = appcoef(c,l,Name,k);
            # To get approximation at level k (cA_k), we perform decomp at level k and take approx.
            # This is cleaner than reconstructing from full decomp manually.
            temp_coeffs = pywt.wavedec(X, Name, level=k)
            a = temp_coeffs[0]
            
            # a = a(ones(1,2^k),:);
            a_up = np.repeat(a, 2**k)
            
            # cA(k,:) = wkeep1(a(:)',len);
            cA[k-1, :] = wkeep1(a_up, len_X)
            
        cA[np.abs(cA) < threshold] = 0
        
        return cD, cA
