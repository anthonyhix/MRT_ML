# This function takes in a wav file and preprocesses it to be used in the model

import numpy as np

def preprocess_wav(y, freq_min = 1000, freq_max = 7000, time_min = 0, time_max = float('inf')):
    # Process the wav file with an FFT
    sr = y[1]
    y = y[0]
    sample_min = int(time_min * sr)
    sample_max = int(np.min([ time_max * sr, len(y)]))
    freq_max = np.min([freq_max, sr])
    out = np.fft.rfft(y[sample_min:sample_max])
    return out[freq_min:freq_max].real
