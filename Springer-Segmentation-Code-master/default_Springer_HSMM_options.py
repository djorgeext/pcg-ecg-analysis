# Python implementation of default_Springer_HSMM_options.m

class SpringerOptions:
    def __init__(self):
        # The sampling frequency at which to extract signal features
        self.audio_Fs = 1000
        
        # The downsampled frequency
        # Set to 50 in Springer paper
        self.audio_segmentation_Fs = 50
        
        # Tolerance for S1 and S2 localization
        self.segmentation_tolerance = 0.1 # seconds
        
        # Whether to use the mex code or not
        self.use_mex = False
        
        # Whether to use the wavelet function or not
        self.include_wavelet_feature = False

def default_Springer_HSMM_options():
    return SpringerOptions()
