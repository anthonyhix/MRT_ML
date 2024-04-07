# This function takes in a wav file and preprocesses it to be used in the model

import numpy as np

def preprocess_wav(y, freq_min = 1000, freq_max = 7000, end_offset = 0.5):
    # Process the wav file with an FFT
    sr = y[1]
    y = y[0]
    group_size = 1
    sample_min = np.max([0, int(end_offset*sr)])
    freq_max = np.min([freq_max, sr])
    #fft = np.fft.rfft(y[sample_min:]).real
    fft = np.fft.rfft(y).real
    out = fft[freq_min:freq_max]
    res = np.sum(out.reshape((freq_max-freq_min)//group_size, group_size), axis = 1).reshape((freq_max-freq_min)//group_size)
    return res
