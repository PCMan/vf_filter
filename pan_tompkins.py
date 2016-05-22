#!/usr/bin/env python3
import pyximport; pyximport.install()  # use Cython
import numpy as np
import scipy as sp
import scipy.signal
# cimport numpy as np  # use cython to speed up numpy array access (http://docs.cython.org/src/userguide/numpy_tutorial.html)
import vf_features


def pan_tompkins_qrs_detection(samples, sampling_rate):
    import matplotlib.pyplot as plt
    """
    Implement Pan-Tompkins QRS detection algorithm:
    * Before calling the function, samples should receive mean subtraction and normalization first.

    References:
    1. Original paper:
        JIAPU PAN AND WILLIS J. TOMPKINS, 1985 A Real-Time QRS Detection Algorithm.
        IEEE TRANSACTIONS ON BIOMEDICAL ENGINEERING, VOL. BME-32, NO.

    2. Detailed explainataions:
        Shital L. Pingale, Nivedita Daimiwal. 2014. Detection of Various Diseases Using ECG Signal in MATLAB.
        International Journal of Recent Technology and Engineering (IJRTE)
        http://www.ijrte.org/attachments/File/v3i1/A1023033114.pdf

    3. Reference implementations:
        http://cnx.org/contents/YR1BUs9_@1/QRS-Detection-Using-Pan-Tompki
        http://www.mathworks.com/matlabcentral/fileexchange/45840-complete-pan-tompkins-implementation-ecg-qrs-detector/content/pan_tompkin.m
    """
    n_samples = len(samples)
    # The original Pan-Tompkins algorithm is for sampling rate 200 Hz.
    if sampling_rate != 200:
        duration = n_samples / sampling_rate
        n_samples = int(duration * 200)
        samples = sp.signal.resample(samples, n_samples)
        sampling_rate = 200
    else:
        duration = n_samples / 200

    fig, axes = plt.subplots(6)
    axes[0].plot(samples)

    # bandpass filtering to maximize energy in 5Hz - 35 Hz
    # low pass filtering
    # y(n) = 2y(n-1) - y(n-2) + x(n) - 2x(n-6) + x(n-12)
    '''
    b = [1, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 1]  # coefficients for x[n], x[n-1],....x[n - len(b)]
    a = [1, -2, 1]  # a[2], a[3], ... a[len(a)] are coefficients for y[n-1], y[n-2], ....y[n - len(a)]
    h = sp.signal.lfilter(b, a, [1] + [0] * 12)  # impulse response
    samples = np.convolve(samples, h)
    '''
    # The convolution stuff looks mysterious. Let's use more stupid but human readable way to implement it.
    x = samples
    y = np.zeros(x.shape)
    for i in range(12, len(x)):
        y[i] = 2 * y[i - 1] - y[i - 2] + x[i] - 2 * x[i - 6] + x[i - 12]
    x = y[12:]
    axes[1].plot(y)

    # high pass filtering
    # y[n] = 32x(n-16) - [y(n-1) + x(n) - x(n-32)]
    '''
    b = [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32, -32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    a = [1, -1]
    h = sp.signal.lfilter(b, a, [1] + [0] * 32)  # impulse response
    samples = np.convolve(samples, h)
    '''
    y = np.zeros(x.shape)
    for i in range(32, len(x)):
        y[i] = 32 * x[i - 16] - (y[i - 1] + x[i] - x[i - 32])
    x = y[32:]
    axes[2].plot(y)

    # differentiation to provide slope information
    # y(n) = 1/8[2x(n) + x(n-1) - x(n-3) - 2x(n-4)]
    '''
    h = np.array([-1, -2, 0, 2, 1]) / 8  # impulse response
    samples = np.convolve(samples, h)
    '''
    y = np.zeros(x.shape)
    for i in range(4, len(x)):
        y[i] = (2 * x[i] + x[i - 1] - x[i - 3] - 2 * x[i - 4]) / 8
    x = y[4:]
    axes[3].plot(y)

    # squaring to make all data points positive and emphasizing the higher frequencies
    # y(n)= x^2 (n)
    x **= 2
    axes[4].plot(x)

    # moving window integration
    '''
    ma_window_size = np.round(0.150 * sampling_rate)
    ma_window = np.ones((1, ma_window_size)) / ma_window_size
    samples = np.convolve(samples, ma_window)
    '''
    y = vf_features.moving_average(x, 30)
    axes[5].plot(y)

    # Fiducial Mark:
    # The QRS complex corresponds to the rising edge of the integration waveform.
    # The time duration of the rising edge is equal to the width of the QRS complex.

    # The higher of the two thresholds in each of the two sets is used for the first analysis of the signal.
    # The lower threshold is used if no QRS is detected in a certain time interval so that a search-back technique is
    # necessary to look back in time for the QRS complex.
    PEAKI = np.max(y)  # overall peak
    SPKI = 0.125 * PEAKI + 0.875 * SPKI  # running estimate of the signal peak
    NPKI = 0.125 * PEAKI + 0.875 * NPKI  # running estimate of the noise peak
    THRESHOLD_Il = NPKI + 0.25 * (SPKI - NPKI)  # first threshold (first step)
    THRESHOLD_I2 = 0.5 * THRESHOLD_Il  # second threshold (for search-back technique)

    plt.show()

    return y


# test the algorithm
if __name__ == '__main__':
    import vf_data
    dataset = vf_data.DataSet()
    for i, sample in enumerate(dataset.get_samples(8.0)):
        signals = sample.signals.astype(dtype='float64')
        signals -= np.mean(signals)
        signals /= np.max(signals)
        ret = pan_tompkins_qrs_detection(signals, sample.info.sampling_rate)
        print(ret)
        if i >= 5:
            break
