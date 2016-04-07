# coding: utf-8
# This implements common VF related features reported in the literatures.
# Felipe AA et al. 2014. Detection of Life-Threatening Arrhythmias Using Feature Selection and Support Vector Machines

import numpy as np
from scipy.signal import butter, lfilter, hamming
# import sampen  # calculate sample entropy
import pyeeg
import matplotlib.pyplot as plt


# time domain/morphology

def threshold_crossing_count(samples, threshold_ratio=0.2):
    threshold = threshold_ratio * np.max(samples)
    n_cross = 0
    higher = samples[0] >= threshold
    first_cross = -1
    last_cross = -1
    for i in range(1, len(samples)):
        sample = samples[i]
        if higher:
            if sample < threshold:
                n_cross += 1
                if first_cross == -1:
                    first_cross = i
                last_cross = i
                higher = False
        else:
            if sample >= threshold:
                n_cross += 1
                if first_cross == -1:
                    first_cross = i
                last_cross = i
                higher = True
    # print threshold, n_cross
    if first_cross == -1:  # no cross at all?
        n_cross = first_cross = last_cross = 0
    return n_cross, first_cross, last_cross


def tcsc_cosine_window(n_samples, sampling_rate):
    quarter = int(0.25 * sampling_rate)  # 1/4 second
    t = np.arange(0, quarter)
    left = 0.5 * (1.0 - np.cos(4 * np.pi * t / sampling_rate))
    t = np.arange(n_samples - quarter, n_samples)
    right = 0.5 * (1.0 - np.cos(4 * np.pi * t / sampling_rate))
    middle = np.ones(n_samples - quarter * 2)
    window = np.concatenate((left, middle, right))
    return window


# average threshold crossing count
# TCSC feature
# Muhammad Abdullah Arafat et al. 2011. A simple time domain algorithm for the detection of
# ventricular fibrillation in electrocardiogram.
def threshold_crossing_sample_counts(samples, n_samples, sampling_rate, window_duration, threshold_ratio=0.2):
    sampling_rate = int(sampling_rate)  # force the use of integer type for sampling rate
    # steps:
    # 1. multiply the samples by a cosine window
    #  w(t) = 1/2(1-cos(4*pi*t)), if 0 <= t <= 1/4 or Ls-1/4 <= t <= Ls
    #  w(t) = 1, otherwise
    # 2. normalize by abs max
    # 3. convert to binary string by comparing abs(x) with threshold V0. (1 if x >= V0); (V0 = 0.2 by default)
    # 4. calculate the samples that cross V0 (number of 1s in the binary sequence)
    #    N = <# of samples that cross V0> / <# of samples> * 100
    # 5. average all Ls-2 N values
    window_size = int(window_duration * sampling_rate)
    window_begin = 0
    window_end = window_size
    tcsc = []
    while window_end <= n_samples:
        # moving window
        window = samples[window_begin:window_end]
        # multiply by a cosine window
        window *= tcsc_cosine_window(window_size, sampling_rate)
        # use absolute values for analysis
        window = np.abs(window)
        # normalize by max
        window /= np.max(window)
        # convert to binary string
        n = np.sum(window > threshold_ratio) * 100.0 / window_size
        tcsc.append(n)
        window_begin += sampling_rate
        window_end += sampling_rate
    # calculate average of all windows
    # print "  TCSC:", tcsc
    return tcsc


# average threshold crossing interval
def threshold_crossing_intervals(samples, n_samples, sampling_rate, threshold_ratio=0.2):
    sampling_rate = int(sampling_rate)  # force the use of integer type for sampling rate
    window_size = int(sampling_rate)  # calculate 1 TCI value per second
    window_begin = 0
    window_end = window_size
    results = []
    while window_end <= n_samples:
        window = samples[window_begin:window_end]
        n_cross, first_cross, last_cross = threshold_crossing_count(window, threshold_ratio)
        results.append((n_cross, first_cross, (window_size - last_cross)))
        window_begin += sampling_rate
        window_end += sampling_rate
    # calculate average TCI of all windows
    # TCI is calculated for every 1 second sub segment, but for each TCI,
    # we also requires some values from its previous and later segments
    tcis = []
    for i in range(1, len(results) - 1):
        n, t2, t3 = results[i]
        t1 = results[i - 1][1]   # last cross of the previous 1-s segment
        t4 = results[i + 1][2]  # first cross of the next 1-s segment
        if n == 0:
            tci = 1000
        else:
            divider = (n - 1 + float(t2)/(t1 + t2) + float(t3)/(t3 + t4))
            tci = float(1000) / divider
        tcis.append(tci)
    # print "  TCIs:", tcis
    return tcis


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


def find_first_local_maximum(x, start=0, threshold=0.0):
    for i in range(start + 1, len(x) - 1):
        if (x[i] - x[i - 1]) >= 0 >= (x[i + 1] - x[i]) and (x[i] >= threshold):
            return i
    return -1


def modified_exponential(samples, sampling_rate, peak_threshold=0.2, time_constant=0.2):
    # similar to standard exponential, but uses relative max, not global max.
    # lift the exponential curve to the relative max again at every intersection.
    # E(t) = Mj * exp(-(t - tm,j) / T), tm,j <= t <= tc,j
    # E(t) = given signal, tc,j <= t <= tm,j+1
    #   Mj: the jth local maximum, tm,j: its time
    #   T: time constant: default to 2.0 second
    #   tc,j: the time value of cross
    n_lifted = 0
    samples = np.array(samples)
    n_samples = len(samples)
    time_constant = time_constant * sampling_rate  # in terms of samples
    # find all local maximum and get their time values
    # FIXME: the original paper does not describe how to correctly identify peaks.
    # Let's set a simple threshold here. :-(
    peak_threshold *= np.max(samples)
    max_time = find_first_local_maximum(samples, start=0, threshold=peak_threshold)
    t = max_time + 1
    # exp_value = list(samples[0:t])
    while t < n_samples:
        sample = samples[t]
        # calculate the exponential value
        local_max = samples[max_time]
        et = local_max * np.exp(-float(t - max_time) / time_constant)
        # exp_value.append(et)
        if et < sample:  # cross happens
            # lift the curve again
            n_lifted += 1
            # find next local maximum
            max_time = find_first_local_maximum(samples, start=t, threshold=peak_threshold)
            if max_time == -1:
                break
            # exp_value.extend(samples[t + 1:max_time + 1])
            t = max_time + 1
        else:
            t += 1

    '''
    plt.plot(samples)
    plt.plot(exp_value)
    plt.show()
    '''
    duration = float(n_samples) / sampling_rate
    return float(n_lifted) / duration


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


# PSR feature:
# 2007. Anton Amann et al. Detecting Ventricular Fibrillation by Time-Delay Methods
# Plotting the time sequence on a phase space plot, and then calculate the boxes
# visited in a 40x40 grid.
def phase_space_reconstruction(samples, sampling_rate, delay=0.5):
    # phase space plotting
    # each data point is: x: x(t), y: x(t + T), where T = 0.5 s by default.
    n_samples = len(samples)
    n_delay = int(delay * sampling_rate)

    x_samples = samples[0:n_samples - n_delay]
    y_samples = samples[n_delay:n_samples]
    offset = np.min(samples)
    axis_range = np.max(samples) - offset

    # convert X and Y values to indices of the 40 x 40 grid
    grid_x = ((x_samples - offset) * 39.0 / axis_range).astype("int")
    grid_y = ((y_samples - offset) * 39.0 / axis_range).astype("int")

    grid = np.zeros((40, 40))
    grid[grid_y, grid_x] = 1
    return float(np.sum(grid)) / 1600


# Bandpass filter:
# http://scipy.github.io/old-wiki/pages/Cookbook/ButterworthBandpass
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y


def butter_lowpass_filter(data, highcut, fs, order=5):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype='low')
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
    # From Computers in Cardiology 2002;29:213−216.
    # This method separates nearly sinusoidal waveforms from the rest.
    # VF is nearly sinusoidal. The idea is to move such signal by half period
    # trying to minimize the sum of the signal and its shifted copy.

    # calculate VF leaks
    # find the central/peak frequency
    # http://cinc.mit.edu/archives/2002/pdf/213.pdf
    if peak_freq != 0:
        cycle = (1.0 / peak_freq)  # in terms of samples
    else:
        cycle = len(samples)
    half_cycle = int(cycle/2)

    original = samples[half_cycle:]
    shifted = samples[:-half_cycle]
    return np.sum(np.abs(original + shifted)) / np.sum(np.abs(original) + np.abs(shifted))


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

    # low pass filter
    # samples = butter_lowpass_filter(samples, 30, sampling_rate)

    # band pass filter
    samples = butter_bandpass_filter(samples, 0.5, 30, sampling_rate)
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
    tcsc = threshold_crossing_sample_counts(samples, len(samples), sampling_rate=sampling_rate, window_duration=3, threshold_ratio=0.1)
    features.append(np.mean(tcsc))
    # features.append(np.std(tcsc))

    # average TCI for every 1-second segments
    tcis = threshold_crossing_intervals(samples, len(samples), sampling_rate=sampling_rate, threshold_ratio=0.1)
    features.append(np.mean(tcis))
    # features.append(np.std(tcis))

    # Standard exponential (STE)
    ste = standard_exponential(samples, sampling_rate)
    features.append(ste)

    # Modified exponential (MEA)
    mea = modified_exponential(samples, sampling_rate)
    features.append(mea)

    # phase space reconstruction (PSR)
    psr = phase_space_reconstruction(samples, sampling_rate)
    features.append(psr)

    # spectral parameters (characteristics of power spectrum)
    # -------------------------------------------------
    # perform discrete Fourier transform
    # apply a hamming window here for side lobe suppression.
    # (the original VF leak paper does not seem to do this).
    # http://www.ni.com/white-paper/4844/en/
    n_samples = len(samples)
    fft = np.fft.fft(samples * hamming(n_samples))
    fft_freq = np.fft.fftfreq(n_samples)

    peak_freq_idx, peak_freq = find_peak_freq(fft, fft_freq)

    # calculate VF leaks
    features.append(vf_leak(samples, peak_freq))

    # calculate other spectral parameters
    # first spectral moment M = 1/peak_freq * (sum(ai * wi)/sum(wi)) for i = 1 to jmax

    # FIXME: what if peak freq = 0?
    if peak_freq == 0:  # is this correct?
        peak_freq = fft_freq[1]

    jmax = min(20 * peak_freq_idx, 100)
    # The original paper approximate the amplitude by real + imaginary parts instead.
    amplitudes = np.abs(fft[0:jmax + 1])
    max_amplitude = np.max(amplitudes)
    # amplitudes whose value is less than 5% of max are set to zero.
    amplitudes[amplitudes < (0.05 * max_amplitude)] = 0
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
    spen = pyeeg.samp_entropy(samples, M=2, R=0.2)
    features.append(spen)

    # MAV
    # mav = mean_absolute_value(samples, sampling_rate)
    # features.append(mav)

    return features
