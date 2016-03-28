# This implements common VF related features reported in the literatures.
# Felipe AA et al. 2014. Detection of Life-Threatening Arrhythmias Using Feature Selection and Support Vector Machines

import numpy as np
from scipy.signal import butter, lfilter
import sampen  # calculate sample entropy
import pyeeg
import matplotlib.pyplot as plt


# time domain/morphology

def threshold_crossing_count(samples, threshold_ratio=0.2):
    threshold = threshold_ratio * np.max(samples)
    n_cross = 0
    higher = samples[0] >= threshold
    for sample in samples:
        if higher:
            if sample < threshold:
                n_cross += 1
                higher = False
        else:
            if sample >= threshold:
                n_cross += 1
                higher = True
    # print threshold, n_cross
    return n_cross


# threshold crossing count
def average_tcsc(samples, n_samples, sampling_rate, window_duration, threshold_ratio=0.2):
    window_size = window_duration * sampling_rate
    window_begin = 0
    window_end = window_size
    tcsc = []
    while window_end <= n_samples:
        window = samples[window_begin:window_end]
        n_cross = threshold_crossing_count(window, threshold_ratio)
        tcsc.append(n_cross)
        window_begin += sampling_rate
        window_end += sampling_rate
    # calculate average of all windows
    return np.mean(tcsc) if tcsc else 0.0


def standard_exponential(samples, sampling_rate, time_constant=3):
    # find the max amplitude in the sample sequence
    # put an exponential like function through this point
    # E(t) = M * exp(- |t-tm| / tc), where
    # M: max signal amplitude at time tm
    # tc: time constant (default to 3)
    max_time = np.argmax(samples)
    max_amplitude = samples[max_time]
    time_constant *= sampling_rate  # convert time from second to samples

    # exp_func(t) = max_amplitude * np.exp(-np.abs(t - max_time) / time_constant)
    # calculate intersections n of this curve with the ECG signal
    n_crosses = 0.0
    higher = True if samples[0] > max_amplitude * np.exp(-np.abs(0 - max_time) / time_constant) else False
    for t in range(1, len(samples) - 1):
        threshold = max_amplitude * np.exp(-np.abs(t - max_time) / time_constant)
        sample = samples[t]
        if higher:
            if sample < threshold:
                higher = False
                n_crosses += 1
        else:
            if sample > threshold:
                higher = True
                n_crosses += 1
    duration = float(len(samples)) / sampling_rate
    return n_crosses / duration


def modified_exponential(samples):
    return 0.0


# Mean absolute value (MAV)
# Emran M Abu Anas et al. 2010. Sequential algorithm for life threatening cardiac pathologies detection based on
# mean signal strength and EMD functions.
# http://biomedical-engineering-online.biomedcentral.com/articles/10.1186/1475-925X-9-43
def mean_absolute_value(samples, sampling_rate, window_duration=2.0):
    # pre-processing: mean subtraction
    # FIXME: the samples we got here already received normalization, which is different from that in the original paper
    n_samples = len(samples)
    mavs = []
    window_size = sampling_rate * window_duration
    window_begin = 0
    window_end = window_size
    # 2-sec moving window with 1 sec step
    while window_end <= n_samples:
        # normalization within the window
        window_samples = samples[window_begin:window_end]
        window_samples /= np.max(window_samples)
        mavs.append(np.mean(window_samples))
        # move to next frame
        window_begin += sampling_rate
        window_end += sampling_rate
    return np.mean(mavs)


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


def moving_average(samples, order=5):
    # https://www.otexts.org/fpp/6/2
    n_samples = len(samples)
    ma = np.zeros(n_samples - order + 1)
    k = int(order / 2)
    for t in range(k, n_samples - k):
        ma[t - k] = np.mean(samples[(t - k):(t + k + 1)])
    return ma


# find the peak frequency and its index in the FFT spectrum
def find_peak_freq(fft, fft_freq):
    peak_freq_idx = fft[:len(fft)/2].argmax()
    return peak_freq_idx, fft_freq[peak_freq_idx]


# the VF leak algorithm
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


# the original sample entropy algorithm is too slow.
# The implementation provided by sampen python package is also slow.
# here we use sample entropy implemented by PyEEG project, which is a little bit faster.
# https://github.com/forrestbao/pyeeg
def sample_entropy(samples, window_size):
    window_begin = 0
    window_end = window_size
    n_samples = len(samples)
    results = []
    while window_end <= n_samples:
        # Ref: Haiyan Li. 2009 Detecting Ventricular Fibrillation by Fast Algorithm of Dynamic Sample Entropy
        # N = 1250 , r = 0.2 x SD, m = 2 worked well for the characterization ECG signals.
        # N = 1250 = 250 Hz x 5 seconds (5-sec window)
        # spens = sampen.sampen2(samples[window_begin:window_end], mm=2)
        # i, entropy, stddev = spens[2]  # unpack the result with m=2
        entropy = pyeeg.samp_entropy(samples[window_begin:window_end], M=2, R=0.2)
        if entropy is None:  # it's possible that sampen2() returns None here
            entropy = 0.0
        window_begin += 360
        window_end += 360
        results.append(entropy)
    return np.mean(results)


# Implement the algorithm described in the paper:
# Xu-Sheng Zhang et al. 1999. Detecting Ventricular Tachycardia and Fibrillation by Complexity Measure
def lz_complexity(samples):
    cn = 0

    # find optimal threshold
    pos_peak = np.max(samples)  # positive peak
    neg_peak = np.min(samples)  # negative peak
    n_samples = len(samples)

    pos_count = np.sum(np.logical_and(0.1 * pos_peak > samples, samples > 0))
    neg_count = np.sum(np.logical_and(0.1 * neg_peak < samples, samples < 0))

    if (pos_count + neg_count) < 0.4 * n_samples:
        threshold = 0.0
    elif pos_count < neg_count:
        threshold = 0.2 * pos_peak
    else:
        threshold = 0.2 * neg_peak

    # make the samples a binary string S based on the threshold
    bin_str = bytearray([1 if b else 0 for b in (samples > threshold)])
    s = bytearray([bin_str[0]])  # S=s1
    q = bytearray([bin_str[1]])  # Q=s2
    for i in range(2, n_samples):
        # SQ concatenation with the last char deleted => SQpi
        sq = s + q[:-1]
        if q in sq:  # Q is a substring of v(SQpi)
            q.append(bin_str[i])
        else:
            cn += 1
            s.extend(q)
            q = bytearray([bin_str[i]])

    # normalization => C(n) = c(n)/b(n), b(n) = n/log2 n
    bn = n_samples / np.log2(n_samples)
    return cn / bn


# extract features from raw sample points of the original ECG signal
def extract_features(samples, sampling_rate, plotting=False):
    features = []
    n_samples = len(samples)
    duration = int(n_samples / sampling_rate)

    # convert to float first for later calculations
    # FIXME: this seems to be a python2 problem?
    if samples.dtype != "float64":
        samples = samples.astype("float64")

    if plotting:
        f, ax = plt.subplots(5, sharex=True)
        ax[0].set_title("before preprocessing")
        ax[0].plot(samples)
    # normalize the input ECG sequence
    samples = (samples - np.min(samples)) / (np.max(samples) - np.min(samples))
    if plotting:
        ax[1].set_title("normaliztion")
        ax[1].plot(samples)

    # perform mean subtraction
    samples = samples - np.mean(samples)
    if plotting:
        ax[2].set_title("mean subtraction")
        ax[2].plot(samples)

    # 5-order moving average
    samples = moving_average(samples, order=5)
    if plotting:
        ax[3].set_title("moving average")
        ax[3].plot(samples)

    # band pass filter
    samples = butter_bandpass_filter(samples, 1, 30, sampling_rate)
    if plotting:
        ax[4].set_title("band pass filter")
        ax[4].plot(samples)
        plt.plot(samples)
        plt.show()

    # Time domain/morphology
    # -------------------------------------------------
    # Threshold crossing interval (TCI) and Threshold crossing sample count (TCSC)

    # get all crossing points, use 20% of maximum as threshold
    # calculate average TCSC using a 3-s window
    # using 3-s moving window
    tcsc = average_tcsc(samples, len(samples), sampling_rate=sampling_rate, window_duration=3, threshold_ratio=0.2)
    features.append(tcsc)

    # Standard exponential (STE)
    ste = standard_exponential(samples, sampling_rate)
    features.append(ste)

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
    # approximate the amplitude by real + imaginary parts
    amplitudes = [np.abs(fft[i].real) + np.abs(fft[i].imag) for i in range(0, jmax + 1)]
    spectral_moment = (1 / peak_freq) * np.sum([amplitudes[i] * fft_freq[i] for i in range(0, jmax + 1)]) / np.sum(amplitudes)
    features.append(spectral_moment)

    # A1: 0 -> min(20 * peak_freq, 100)
    # A2: 0.5 -> peak_freq / 2
    # A3: 0.6 bands around 2 * peak_freq -> 8 * peak_freq divided by 0.5 -> min(20 * peak_freq, 100)

    # complexity parameters
    # -------------------------------------------------
    lzc = lz_complexity(samples)
    features.append(lzc)

    # sample entropy (SpEn)
    # spen = sample_entropy(samples, 5 * sampling_rate)
    # spen = pyeeg.samp_entropy(samples, M=2, R=0.2)
    # features.append(spen)

    # MAV
    # mav = mean_absolute_value(samples, sampling_rate)
    # features.append(mav)

    return features
