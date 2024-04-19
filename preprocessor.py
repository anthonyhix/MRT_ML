# This function takes in a wav file and preprocesses it to be used in the model

import scipy as sp
import numpy as np

def preprocess_wav(y, freq_min = 1000, freq_max = 7000, sample_time = float('inf'), group_size = 1):
    # # Process the wav file with an FFT using numpy
    # sr = y[1]
    # y = y[0]
    # sample_min = int(start_offset * sr)
    # sample_max = int(len(y) - end_offset * sr)
    # freq_max = np.min([freq_max, sr])
    # y = y[sample_min:sample_max]
    # fft = np.fft.rfft(y).real
    # out = fft[freq_min:freq_max]
    # out_grouped = np.sum(out[0:out.shape[0] - out.shape[0] % group_size].reshape(out.shape[0]//group_size, group_size), axis = 1)
    # Process the wav file with an stft using scipy
    sr = y[1]
    y = y[0]
    sample_min = int(len(y) - sample_time * sr)
    sample_max = len(y)
    freq_max = np.min([freq_max, sr])
    y = y[sample_min:sample_max]
    f, t, Zxx = sp.signal.stft(y)
    print('Data Loaded')
    return Zxx.real
    