# This function takes in a wav file and preprocesses it to be used in the model

import numpy as np

def preprocess_wav(y, freq_min = 1000, freq_max = 7000):
    # Process the wav file with an FFT
    freq_max = np.min([freq_max, y[1]])
    out = np.fft.fft(y[0])
    return out[freq_min:freq_max].real
