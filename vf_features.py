# This implements common VF related features reported in the literatures.
# Felipe AA et al. 2014. Detection of Life-Threatening Arrhythmias Using Feature Selection and Support Vector Machines

import numpy as np
from sklearn import preprocessing
from scipy.signal import butter, lfilter


# time domain/morphology
# get the index of threshold crossing points in the segment
def threshold_crossing(samples, threshold):
    crossing = []
    high = samples[0] >= threshold
    for i, sample in enumerate(samples):
        if high:
            if sample < threshold:
                crossing.append(i)
                high = False
        else:
            if sample >= threshold:
                crossing.append(i)
                high = True
    return crossing


def average_tci(crossing, n_samples, sampling_rate):
    window_size = 3 * sampling_rate
    window_begin = 0
    window_end = window_size
    tcis = []
    n_crossing = 0
    for i, crossing_idx in enumerate(crossing):
        if crossing_idx >= window_end:
            # end of the current window and begin of the next window
            window_end += window_size
            if window_end > n_samples:
                break
            window_begin += window_size
            # calculate TCI
            t1 = crossing[i - 1]
            t2 = 0
            t3 = 0
            t4 = crossing[i + 1]
            tci = 1000/((n_crossing - 1) + t2 / (t1 + t2) + t3 / (t3 + t4))
            tcis.append(tci)
            n_crossing = 0
        n_crossing += 1
    # calculate average of all windows
    return np.mean(tcis) if tcis else 0.0


def average_tcsc(crossing, n_samples, window_size):
    window_begin = 0
    window_end = window_size
    tcsc = []
    n_crossing = 0
    for i, crossing_idx in enumerate(crossing):
        if crossing_idx >= window_end:
            # end of the current window and begin of the next window
            window_end += window_size
            if window_end > n_samples:
                break
            window_begin += window_size
            tcsc.append(n_crossing)
            n_crossing = 0
        n_crossing += 1
    # calculate average of all windows
    return np.mean(tcsc) if tcsc else 0.0


def standard_exponential(samples):
    pass

def modified_exponential(samples):
    pass


# spectral parameters

def vf_filter(samples):
    pass

def spectral_algorithm(samples):
    pass

def median_freq(samples):
    pass


# Bandpass filter:
# http://scipy.github.io/old-wiki/pages/Cookbook/ButterworthBandpass
def butter_bandpass(lowcut, highcut, fs, order=5):
   nyq = 0.5 * fs
   low = lowcut / nyq
   high = highcut / nyq
   b, a = butter(order, [low, high], btype='band')
   return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
   b, a = butter_bandpass(lowcut, highcut, fs, order=order)
   y = lfilter(b, a, data)
   return y


# extract features from raw sample points of the original ECG signal
def extract_features(samples, sampling_rate):
    # normalize the input ECG sequence
    samples = (samples - np.min(samples)) / (np.max(samples) - np.min(samples))

    # band pass filter
    samples = butter_bandpass_filter(samples, 1, 30, sampling_rate)
    
    features = []
    n_samples = len(samples)
    duration = int(n_samples / sampling_rate)
    # Time domain/morphology
    # -------------------------------------------------
    # Threshold crossing interval (TCI) and Threshold crossing sample count (TCSC)
    # get all crossing points
    crossing = threshold_crossing(samples, threshold=0.2)
    # calculate average TCI and TCSC using a 3-s window
    # using 3-s moving window
    window_size = 3 * sampling_rate
    window_begin = 0
    window_end = window_size
    tci = []
    tcsc = []
    n_crossing = 0
    for i, crossing_idx in enumerate(crossing):
        if crossing_idx >= window_end:
            # end of the current window and begin of the next window
            window_end += window_size
            if window_end > n_samples:
                break
            window_begin += window_size
            tcsc.append(n_crossing)
            # calculate TCI
            t1 = crossing[i - 1]

            n_crossing = 0
        n_crossing += 1
    # calculate average of all windows
    # features.append(np.mean(tci) if tci else 0.0)
    features.append(np.mean(tcsc) if tcsc else 0.0)

    # Standard exponential (STE)

    # Modified exponential (MEA)

    # Mean absolute value (MAV) of 2-s segments
    '''
    pos = 0
    mav = 0.0
    window_size = 2 * sampling_rate
    n_windows = int(n_samples / window_size)
    while (pos + window_size) < n_samples:
        window_samples = samples[pos : pos + window_size]
        mav += np.mean(window_samples)
    mav /= n_windows
    features.append(mav)
    '''

    # spectral parameters
    # -------------------------------------------------

    # complexity parameters
    # -------------------------------------------------
    return features
