# This implements common VF related features reported in the literatures.
# Felipe AA et al. 2014. Detection of Life-Threatening Arrhythmias Using Feature Selection and Support Vector Machines

import numpy as np
from sklearn import preprocessing
from scipy.signal import butter, lfilter
import sampen  # calculate sample entropy

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


def find_peak_freq(fft, fft_freq):
    peak_freq_idx = fft[:len(fft)/2].argmax()
    return peak_freq_idx, fft_freq[peak_freq_idx]


def vf_leak(samples, peak_freq):
    # calculate VF leaks
    # find the central/peak frequency
    # http://cinc.mit.edu/archives/2002/pdf/213.pdf
    # T = (1/f) * sample_rate
    if peak_freq != 0:
        cycle = (1 / peak_freq)  # in terms of samples
    else:
        cycle = len(samples)  # FIXME: should we use infinity here?

    vf_leak_numerator = 0.0
    vf_leak_denominator = 0.0
    half_cycle = int(cycle/2)
    for i in range(half_cycle, len(samples)):
        vf_leak_numerator += np.abs(samples[i] + samples[i - half_cycle])
        vf_leak_denominator += np.abs(samples[i]) + np.abs(samples[i - half_cycle])
    return vf_leak_numerator / vf_leak_denominator


def sample_entropy(samples, window_size):
    window_begin = 0
    window_end = window_size
    n_samples = len(samples)
    results = []
    while window_end < n_samples:
        # Ref: Haiyan Li. 2009 Detecting Ventricular Fibrillation by Fast Algorithm of Dynamic Sample Entropy
        # N = 1250 , r = 0.2 x SD, m = 2 worked well for the characterization ECG signals.
        # N = 1250 = 250 Hz x 5 seconds (5-sec window)
        spens = sampen.sampen2(samples[window_begin:window_end], mm=2)
        i, entropy, stddev = spens[2]  # unpack the result with m=2
        if entropy is None:  # it's possible that sampen2() returns None here
            entropy = 0.0
        window_begin += window_size
        window_end += window_size
        results.append(entropy)
    return np.mean(results)


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
    # calculate average TCSC using a 3-s window
    # using 3-s moving window
    features.append(average_tcsc(crossing, len(samples), window_size=3 * sampling_rate))

    # Standard exponential (STE)

    # Modified exponential (MEA)

    # spectral parameters (characteristics of power spectrum)
    # -------------------------------------------------

    # perform discrete Fourier transform
    fft = np.fft.fft(samples)
    fft_freq = np.fft.fftfreq(len(samples))

    peak_freq_idx, peak_freq = find_peak_freq(fft, fft_freq)

    # calculate VF leaks
    features.append(vf_leak(samples, peak_freq))

    # calculate other spectral parameters
    # first spectral moment M = 1/peak_freq * (sum(ai * wi)/sum(wi)) for i = 1 to jmax
    # FIXME: what if peak freq = 0?
    if peak_freq == 0:  # is this correct?
        peak_freq = fft_freq[1]

    jmax = min(20 * peak_freq_idx, 100)
    # approximate the amplitude by real + imag parts
    amplitudes = [np.abs(fft[i].real) + np.abs(fft[i].imag) for i in range(0, jmax + 1)]
    spectral_moment = (1 / peak_freq) * np.sum([amplitudes[i] * fft_freq[i] for i in range(0, jmax + 1)]) / np.sum(amplitudes)
    features.append(spectral_moment)

    # A1: 0 -> min(20 * peak_freq, 100)
    # A2: 0.5 -> peak_freq / 2
    # A3: 0.6 bands around 2 * peak_freq -> 8 * peak_freq divided by 0.5 -> min(20 * peak_freq, 100)

    # complexity parameters
    # -------------------------------------------------

    # sample entropy (SpEn)
    features.append(sample_entropy(samples, 5 * sampling_rate))
    print features

    return features
