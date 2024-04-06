# This function takes in a wav file and preprocesses it to be used in the model

import numpy as np

def preprocess_wav(y, freq_min = 1000, freq_max = 7000, end_offset = 5):
    # Process the wav file with an FFT
    sr = y[1]
    y = y[0]
    sample_min = np.min([0, int(end_offset*sr)])
    freq_max = np.min([freq_max, sr])
    out = np.fft.rfft(y[sample_min:])
    return out[freq_min:freq_max].real
