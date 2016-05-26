#!/usr/bin/env python3
# This implements common VF related features reported in the literatures.
# Felipe AA et al. 2014. Detection of Life-Threatening Arrhythmias Using Feature Selection and Support Vector Machines
import pyximport; pyximport.install()  # use Cython
import numpy as np
cimport numpy as np  # use cython to speed up numpy array access (http://docs.cython.org/src/userguide/numpy_tutorial.html)
import scipy.signal as signal
import pyeeg  # calculate sample entropy
import matplotlib.pyplot as plt
from array import array  # python array with static types
import pickle
from qrs_detect import qrs_detect  # QRS detector
from ptsa.ptsa.emd import emd  # empirical mode decomposition
import sys  # for byteorder


feature_names = (
    "TCSC",             # threshold crossing sample count
    "TCI",              # threshold crossing interval
    "STE",              # standard exponential
    "MEA",              # modified exponential
    "PSR",              # phase space reconstruction (time-delay method)
    "HILB",             # Hilbert transformation-based PSR
    "VF",               # VF leak (VF filter)
    "M",                # first spectral moment
    "A2",               # A2 of spectrum
    "FM",               # central frequency of power spectrum
    "LZ",               # LZ complexity
    "SpEn",             # sample entropy (5-second)
    "MAV",              # mean absolute value
    "Count1",           # count 1
    "Count2",           # count 2
    "Count3",           # count 3
    "Amplitude",        # amplitude of the signals
    "IMF1_LZ",          # LZ complexity of EMD IMF1
    "IMF5_LZ",          # LZ complexity of EMD IMF5
    "RR",               # average RR interval (0 if no QRS is detected)
    "RR_Std",           # standard deviation of RR (0 if no QRS is detected)
    "RR_CV",            # standard deviation of RR / mean RR (0 if no QRS is detected)
    "UR",               # unknown beats/all beats (0 if no QRS is detected)
    "VR",               # VPC beats/all beats (0 if no QRS is detected)
    # "SpecLZ",    # LZ complexity of spectrum
)


feature_names_set = set(feature_names)

# time domain/morphology

cdef tuple threshold_crossing_count(np.ndarray[double, ndim=1] samples, double threshold_ratio=0.2):
    cdef double threshold = threshold_ratio * np.max(samples)
    cdef int n_cross = 0
    cdef bint higher = samples[0] >= threshold
    cdef int first_cross = -1
    cdef int last_cross = -1
    cdef double sample
    cdef int i
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


# average threshold crossing count
# TCSC feature
# Muhammad Abdullah Arafat et al. 2011. A simple time domain algorithm for the detection of
# ventricular fibrillation in electrocardiogram.
cdef double threshold_crossing_sample_counts(np.ndarray[double, ndim=1] samples, int sampling_rate, double window_duration=3.0, double threshold_ratio=0.2):
    # steps:
    # 1. multiply the samples by a cosine window
    #  w(t) = 1/2(1-cos(4*pi*t)), if 0 <= t <= 1/4 or Ls-1/4 <= t <= Ls
    #  w(t) = 1, otherwise
    # 2. normalize by abs max
    # 3. convert to binary string by comparing abs(x) with threshold V0. (1 if x >= V0); (V0 = 0.2 by default)
    # 4. calculate the samples that cross V0 (number of 1s in the binary sequence)
    #    N = <# of samples that cross V0> / <# of samples> * 100
    # 5. average all Ls-2 N values
    cdef int n_samples = len(samples)
    cdef int window_size = int(window_duration * sampling_rate)
    cdef int window_begin = 0
    cdef int window_end = window_size
    if window_end > n_samples:
        window_end = window_size = n_samples

    tcsc = array("d")
    cdef double n = 0
    cdef np.ndarray[double, ndim=1] window
    while window_end <= n_samples:
        # moving window
        window = samples[window_begin:window_end]

        # multiply by a cosine window (Tukey window)
        # window *= tcsc_cosine_window(window_size, sampling_rate)
        window *= signal.tukey(window_size, alpha=(0.5 / window_duration))

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
    return np.mean(tcsc)


# average threshold crossing interval
cdef double threshold_crossing_intervals(np.ndarray[double, ndim=1] samples, int sampling_rate, double threshold_ratio=0.2):
    cdef int n_samples = len(samples)
    cdef int window_size = int(sampling_rate)  # calculate 1 TCI value per second
    cdef int window_begin = 0
    cdef int window_end = window_size
    if window_end > n_samples:
        window_end = window_size = n_samples

    cdef list results = []
    cdef int n_cross, first_cross, last_cross
    cdef np.ndarray[double, ndim=1] window
    while window_end <= n_samples:
        window = samples[window_begin:window_end]
        n_cross, first_cross, last_cross = threshold_crossing_count(window, threshold_ratio)
        results.append((n_cross, first_cross, (window_size - last_cross)))
        window_begin += sampling_rate
        window_end += sampling_rate
    # calculate average TCI of all windows
    # TCI is calculated for every 1 second sub segment, but for each TCI,
    # we also requires some values from its previous and later segments
    tcis = array("d")
    cdef int first_result = 1
    cdef int last_result = len(results) - 1

    if len(results) < 3:  # special handling for less than 3 results
        first_result = 0
        last_result = 2

    cdef double n, t1, t2, t3, t4, divider, tci
    cdef int i
    for i in range(first_result, last_result):
        n, t2, t3 = results[i]
        t1 = results[i - 1][1] if i >= 1 else 0  # last cross of the previous 1-s segment
        t4 = results[i + 1][2] if (i + 1) < len(results) else 0  # first cross of the next 1-s segment
        if n == 0:
            tci = 1000
        else:
            divider = (n - 1 + t2 / (t1 + t2) + t3 / (t3 + t4))
            tci = 1000 / divider
        tcis.append(tci)
    # print "  TCIs:", tcis
    return np.mean(tcis)


cdef double standard_exponential(samples, int sampling_rate, int time_constant=3):
    # find the max amplitude in the sample sequence
    # put an exponential like function through this point
    # E(t) = M * exp(- |t-tm| / tc), where
    # M: max signal amplitude at time tm
    # tc: time constant (default to 3)
    cdef int max_time = np.argmax(samples)
    cdef double max_amplitude = samples[max_time]
    time_constant *= sampling_rate  # convert time from second to samples

    # exp_func(t) = max_amplitude * np.exp(-np.abs(t - max_time) / time_constant)
    # calculate intersections n of this curve with the ECG signal
    cdef double n_crosses = 0.0
    cdef bint higher = True if samples[0] > max_amplitude * np.exp(-np.abs(0 - max_time) / time_constant) else False
    cdef double threshold, sample
    cdef int t
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
    cdef double duration = float(len(samples)) / sampling_rate
    return n_crosses / duration


cdef double modified_exponential(np.ndarray[double, ndim=1] samples, int sampling_rate, double time_constant=0.2):
    # similar to standard exponential, but uses relative max, not global max.
    # lift the exponential curve to the relative max again at every intersection.
    # E(t) = Mj * exp(-(t - tm,j) / T), tm,j <= t <= tc,j
    # E(t) = given signal, tc,j <= t <= tm,j+1
    #   Mj: the jth local maximum, tm,j: its time
    #   T: time constant: default to 2.0 second
    #   tc,j: the time value of cross
    cdef int n_lifted = 0
    cdef int n_samples = len(samples)
    time_constant *= sampling_rate  # in terms of samples
    # find all local maximum and get their time values (the original paper did not describe how to do this. :-( )
    cdef int half_peak_width = int(np.round(0.05 * sampling_rate))  # make peak detection less sensitive
    local_max_indices = iter(signal.argrelmax(samples, order=half_peak_width)[0])
    cdef double sample, local_max
    cdef int t, local_max_idx
    try:
        local_max_idx = next(local_max_indices)
        local_max = samples[local_max_idx]
        t = local_max_idx + 1
        while t < n_samples:
            sample = samples[t]
            # calculate the exponential value
            et = local_max * np.exp(-(t - local_max_idx) / time_constant)
            if et < sample:  # cross happens
                # find next local maximum
                while True:
                    local_max_idx = next(local_max_indices)
                    if local_max_idx > t:
                        break
                # lift the curve again at the next local maximum
                n_lifted += 1
                local_max = samples[local_max_idx]
                t = local_max_idx + 1
            else:
                t += 1
    except StopIteration:  # no more local maximum values
        pass
    cdef double duration = float(n_samples) / sampling_rate
    return n_lifted / duration


# Mean absolute value (MAV)
# Emran M Abu Anas et al. 2010. Sequential algorithm for life threatening cardiac pathologies detection based on
# mean signal strength and EMD functions.
# http://biomedical-engineering-online.biomedcentral.com/articles/10.1186/1475-925X-9-43
cdef double mean_absolute_value(np.ndarray[double, ndim=1] samples, int sampling_rate, double window_duration=2.0):
    # pre-processing: mean subtraction
    cdef int n_samples = len(samples)
    mavs = array("d")
    cdef int window_size = int(sampling_rate * window_duration)
    cdef int window_begin = 0
    cdef int window_end = window_size
    # 2-sec moving window with 1 sec step
    cdef np.ndarray[double, ndim=1] window_samples
    while window_end <= n_samples:
        # normalization within the window
        window_samples = samples[window_begin:window_end]
        # get the absolute values of the signals, and then normalize using max abs value.
        window_samples = np.abs(window_samples)
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
cdef double phase_space_reconstruction(np.ndarray[double, ndim=1] samples, int sampling_rate, double delay=0.5):
    # phase space plotting
    # each data point is: x: x(t), y: x(t + T), where T = 0.5 s by default.
    cdef int n_samples = len(samples)
    cdef int n_delay = int(delay * sampling_rate)

    cdef np.ndarray[double, ndim=1] x_samples = samples[0:n_samples - n_delay]
    cdef np.ndarray[double, ndim=1] y_samples = samples[n_delay:n_samples]
    cdef double offset = np.min(samples)
    cdef double axis_range = np.max(samples) - offset

    # convert X and Y values to indices of the 40 x 40 grid
    cdef np.ndarray[np.int8_t, ndim=1] grid_x = ((x_samples - offset) * 39.0 / axis_range).astype(np.int8)
    cdef np.ndarray[np.int8_t, ndim=1] grid_y = ((y_samples - offset) * 39.0 / axis_range).astype(np.int8)

    cdef np.ndarray[np.int8_t, ndim=2] grid = np.zeros((40, 40), dtype=np.int8)
    grid[grid_y, grid_x] = 1
    return np.sum(grid) / 1600


# HILB feature based on Hilbert transformation + phase space plot
# A Amann et al. 2005. A New Ventricular Fibrillation Detection Algorithm for Automated External Defibrillators
# Computers in Cardiology 2005;32:559−562.
cdef double hilbert_psr(np.ndarray[double, ndim=1] samples, int sampling_rate):
    # each data point is: x: x(t), y: xH(t), where xH(t) is the Hilbert transform of x(t)
    cdef float duration = len(samples) / sampling_rate
    # down sample to 50 Hz for fast computation
    cdef int n_samples = int(duration * 50)

    # phase space plotting (40 x 40 grid)
    cdef np.ndarray[double, ndim=1] x_samples = signal.resample(samples, n_samples)
    cdef np.ndarray[double complex, ndim = 1] analytical_signals = signal.hilbert(x_samples)
    cdef np.ndarray[double, ndim = 1] y_samples = np.imag(analytical_signals)  # the imaginary part of the analytical signal is the Hilbert transform of X(t)

    cdef double x_offset = np.min(x_samples)
    cdef double x_range = np.max(x_samples) - x_offset
    cdef double y_offset = np.min(y_samples)
    cdef double y_range = np.max(y_samples) - y_offset

    # convert X and Y values to indices of the 40 x 40 grid
    cdef np.ndarray[np.int8_t, ndim=1] grid_x = ((x_samples - x_offset) * 39.0 / x_range).astype(np.int8)
    cdef np.ndarray[np.int8_t, ndim=1] grid_y = ((y_samples - y_offset) * 39.0 / y_range).astype(np.int8)

    # calculate the number of visited cells
    cdef np.ndarray[np.int8_t, ndim=2] grid = np.zeros((40, 40), dtype=np.int8)
    grid[grid_y, grid_x] = 1
    return np.sum(grid) / 1600


cpdef np.ndarray[double, ndim=1] drift_supression(np.ndarray[double, ndim=1] data, double cutoff_freq, double sampling_rate):
    # low pass filter for drift supression
    # Reference: https://homepages.fhv.at/ku/karl/VF/filtering.m
    cdef double T = 1.0 / sampling_rate  # sampling peroid [s]
    cdef double c1 = 1.0 / (1.0 + np.tan(cutoff_freq * np.pi * T))
    cdef double c2 = (1.0 - np.tan(cutoff_freq * np.pi * T)) / (1 + np.tan(cutoff_freq * np.pi * T))
    cdef list b = [c1, -c1]
    cdef list a = [1.0, -c2]
    return signal.filtfilt(b, a, data)


# Butterworth filter:
# http://scipy.github.io/old-wiki/pages/Cookbook/ButterworthBandpass
cpdef np.ndarray[double, ndim=1] butter_lowpass_filter(np.ndarray[double, ndim=1] data, double highcut_freq, double fs, int order=5):
    cdef double nyq = 0.5 * fs
    cdef double high = highcut_freq / nyq
    cdef np.ndarray[double, ndim = 1] b, a
    b, a = signal.butter(order, high, btype="lowpass")
    cdef np.ndarray[double, ndim=1] y = signal.filtfilt(b, a, data)  # zero phase filter (no phase distortion)
    return y


cpdef np.ndarray[double, ndim=1] moving_average(np.ndarray[double, ndim=1] samples, int order=5):
    """
    # Reference: https://homepages.fhv.at/ku/karl/VF/filtering.m
    cdef np.ndarray[double, ndim = 1] b, a
    b = np.ones(order) / order
    a = np.ones(1)
    return signal.lfilter(b, a, samples)
    """
    # Reference: https://www.otexts.org/fpp/6/2
    # Moving average can be calculated using convolution
    # http://matlabtricks.com/post-11/moving-average-by-convolution
    return np.convolve(samples, np.ones(order) / order, mode="same")


# the VF leak algorithm
cdef double vf_leak(np.ndarray[double, ndim=1] samples, np.ndarray[double complex, ndim=1] fft, np.ndarray[double, ndim=1] fft_freq):
    # This method separates nearly sinusoidal waveforms from the rest.
    # VF is nearly sinusoidal. The idea is to move such signal by half period
    # trying to minimize the sum of the signal and its shifted copy.

    # calculate VF leaks
    # find the central/peak frequency
    # http://cinc.mit.edu/archives/2002/pdf/213.pdf
    cdef int peak_freq_idx = np.argmax(fft)
    cdef double peak_freq = fft_freq[peak_freq_idx]  # From Computers in Cardiology 2002;29:213−216.

    cdef double cycle
    if peak_freq != 0:
        cycle = (1.0 / peak_freq)  # in terms of samples
    else:
        cycle = len(samples)
    cdef double half_cycle = int(cycle/2)

    cdef np.ndarray[double, ndim=1] original = samples[half_cycle:]
    cdef np.ndarray[double, ndim=1] shifted = samples[:-half_cycle]
    return np.sum(np.abs(original + shifted)) / np.sum(np.abs(original) + np.abs(shifted))


# spectral parameters (M and A2)
cpdef tuple spectral_features(np.ndarray[double complex, ndim=1] fft, np.ndarray[double, ndim=1] fft_freq, int sampling_rate, plot=None):
    # Reference: BioMedical Engineering OnLine 2005, 4:60
    # The ECG of most normal heart rhythms is a broadband signal with major harmonics up to about 25 Hz.
    # During VF, the ECG becomes concentrated in a band of frequencies between 3 and 10 Hz, with particularly
    # low frequencies of undercooled victims.

    # Find the peak frequency within the range of 0.5 Hz - 9.0 Hz
    # NOTE: the unit of time is sample number, so the unit of frequency is not Hz here
    # to convert to Hz, we have to multiply the frequencies with sample rate.
    # fft_freq_hz = fft_freq * sampling_rate

    # The original paper approximate the amplitude by real + imaginary parts instead of true modulus.
    cdef np.ndarray[double, ndim=1] amplitudes = np.abs(fft)

    cdef int min_freq_idx = np.searchsorted(fft_freq, 0.5 / sampling_rate, side="right")
    cdef int max_freq_idx = np.searchsorted(fft_freq, 9.0 / sampling_rate, side="left")
    # Here the peak_freq_index should + the index of 0.5 Hz
    cdef int peak_freq_idx = np.argmax(amplitudes[min_freq_idx:max_freq_idx]) + min_freq_idx
    cdef double peak_freq = fft_freq[peak_freq_idx]
    cdef double peak_amplitude = amplitudes[peak_freq_idx]

    # amplitudes whose value is less than 5% of peak amplitude are set to zero.
    amplitudes[amplitudes < (0.05 * peak_amplitude)] = 0

    # first spectral moment M = 1/peak_freq * (ai dot wi/sum(ai) for i = 1 to jmax
    # jmax: the index of the highest investigated frequency
    # peak_freq: the frequency of the component with the largest amplitude in the range 0.5 - 9 Hz
    #            amplitudes whose value < 5% of peak_freq are set to 0.
    cdef double spectral_moment = 0.0
    cdef double spec_max_freq = min(20 * peak_freq, 100.0 / sampling_rate)  # convert 100 Hz to use sample count as time unit
    cdef int max_spec_idx = np.searchsorted(fft_freq, spec_max_freq, side="left")
    cdef np.ndarray[double, ndim=1] m_amplitudes = amplitudes[0:max_spec_idx]
    cdef double sum_m_amplitudes = np.sum(m_amplitudes)

    if sum_m_amplitudes != 0:  # in theory, division by zero should not happen here
        spectral_moment = (1 / peak_freq) * np.dot(m_amplitudes, fft_freq[0:max_spec_idx]) / sum_m_amplitudes

    # calculate A2
    # frequency range: 0.7 * peak_freq - 1.4 * peak_freq
    cdef double a2 = 0.0
    cdef int min_a2_idx = np.searchsorted(fft_freq, 0.7 * peak_freq, side="right")
    cdef int max_a2_idx = np.searchsorted(fft_freq, 1.4 * peak_freq, side="left")
    cdef double sum_amplitudes = np.sum(amplitudes[0:max_spec_idx])
    if sum_amplitudes != 0:  # in theory, division by zero should not happen here
        a2 = np.sum(amplitudes[min_a2_idx:max_a2_idx]) / sum_amplitudes

    # plot on a matplotlib.pyplot object
    if plot:
        plot.set_title("Spectral parameters")
        plot.plot(fft_freq * sampling_rate, amplitudes, color="k")
        plot.axvline(x=spec_max_freq * sampling_rate, color="r")
        plot.axvline(x=peak_freq * sampling_rate, color="r")
        plot.axvline(x=spec_max_freq * sampling_rate, color="b")
        plot.axvline(x=fft_freq[min_a2_idx] * sampling_rate, color="g")
        plot.axvline(x=fft_freq[max_a2_idx] * sampling_rate, color="g")

    return (spectral_moment, a2)


# FM feature: central frequency that biset the power spectrum
# IEEE TRANSACTIONS ON BIOMEDICAL ENGINEERING, VOL 37, NO 6. JUNE 1990
# The Median Frequency of the ECG During Ventricular Fibrillation: Its Use in an Algorithm for Estimating the Duration of Cardiac Arrest.
cdef double central_frequency(np.ndarray[double complex, ndim=1] fft, np.ndarray[double, ndim=1] fft_freq, int sampling_rate):
    # FM = sum(fi * pi) / sum(pi)
    #   pi: power component at ith frequency
    cdef np.ndarray[double, ndim = 1] power_spec = np.abs(fft) ** 2
    cdef double central_freq = np.dot(fft_freq, power_spec) / np.sum(power_spec)
    # convert the frequency to Hz
    return central_freq * sampling_rate


# the original sample entropy algorithm is too slow.
# The implementation provided by sampen python package is also slow.
# here we use sample entropy implemented by PyEEG project, which is a little bit faster.
# https://github.com/forrestbao/pyeeg
cdef double sample_entropy(np.ndarray[double, ndim=1] samples):
    # Ref: Haiyan Li. 2009 Detecting Ventricular Fibrillation by Fast Algorithm of Dynamic Sample Entropy
    # N = 1250 , r = 0.2 x SD, m = 2 worked well for the characterization ECG signals.
    # N = 1250 = 250 Hz x 5 seconds (5-sec window)
    cdef np.ndarray[double, ndim = 1] spen_samples = samples[-1250:]
    cdef double sample_sd = np.std(spen_samples)
    cdef double spen = pyeeg.samp_entropy(spen_samples, M=2, R=(0.2 * sample_sd))
    return spen


cdef double lempel_ziv_complexity(bytearray bin_str):
    cdef int cn = 0
    cdef int n = len(bin_str)
    cdef bytearray q = bytearray()
    cdef int i, s_end = 1
    for i in range(1, n):
        q.append(bin_str[i])  # append current s[i] to Q
        # SQpi: SQ concatenation with the last char deleted => S + Q[:-1]
        sq_end = s_end + len(q) - 1  # end position of S + Q[:-1]
        if bin_str.find(q, 0, sq_end) == -1:  # Q is NOT a substring of S + Q[:-1]
            cn += 1  # increase complexity
            s_end += len(q)  # append Q to S
            q.clear()  # reset Q

    # normalization => C(n) = c(n)/b(n), b(n) = n/log2 n
    cdef double bn = n / np.log2(n)
    return cn / bn


# Convert an array with supported element types to binary string.
# The result is stored in bytearray with each byte representing a bit in the original data (1 or 0).
# Though this is not memory efficient, it's easier for complexity computation later.
# data should be an Python array.array instance.
cpdef bytearray array_to_binary_byte_string(object data, int bits_per_element, bint little_endian):
    cdef bytearray bin_str = bytearray()
    cdef int byte
    cdef str bit, bits
    cdef int zero_char = ord("0")
    cdef bit_format_spec = "0{0}b".format(bits_per_element)
    for byte in data.tobytes():
        bits = format(byte, bit_format_spec)  # convert to binary
        if little_endian:  # little endian byte order
            bits = bits[-bits_per_element:]  # we want to low bits
        else:   # big endian
            bits = bits[:bits_per_element]  # we want to high bits
        for bit in bits:
            bin_str.append(ord(bit) - zero_char)  # convert from "1" or "0" to integer value
    return bin_str


# Implement the algorithm described in the paper:
# Xu-Sheng Zhang et al. 1999. Detecting Ventricular Tachycardia and Fibrillation by Complexity Measure
cdef double complexity_measure(np.ndarray[double, ndim=1] samples):
    cdef int cn = 0

    # find optimal threshold
    cdef double pos_peak = np.max(samples)  # positive peak
    cdef double neg_peak = np.min(samples)  # negative peak
    cdef int n_samples = len(samples)

    cdef int pos_count = np.sum(np.logical_and(0.1 * pos_peak > samples, samples > 0))
    cdef int neg_count = np.sum(np.logical_and(0.1 * neg_peak < samples, samples < 0))

    cdef double threshold = 0.0
    if (pos_count + neg_count) < 0.4 * n_samples:
        threshold = 0.0
    elif pos_count < neg_count:
        threshold = 0.2 * pos_peak
    else:
        threshold = 0.2 * neg_peak

    # make the samples a binary string S based on the threshold
    cdef bint b
    bin_str = bytearray([1 if b else 0 for b in (samples > threshold)])
    return lempel_ziv_complexity(bin_str)


# Bandpass Filter and Auxiliary Counts (count1, 2, 3)
cdef tuple auxiliary_counts(np.ndarray[double, ndim=1] samples, int sampling_rate):
    # digital filter with central frequency at 14.6 Hz and bandwidth 
    # from 13 to 16.5 Hz (–3dB) is applied onthe ECG data.
    # With a sampling frequency of 250 Hz, the filter is designed by
    # FS[i] = (14 FS[i-1] - 7 FS[i-2] + (S[i] - S[i-2])/2) / 8
    # resample to 250 Hz as needed
    cdef int count1 = 0, count2 = 0, count3 = 0
    cdef int n_samples = len(samples)
    if sampling_rate != 250:
        samples = signal.resample(samples, int((n_samples / sampling_rate) * 250))
        n_samples = len(samples)
    
    # low pass filter
    cdef np.ndarray[double, ndim=1] fs = np.zeros(n_samples)
    cdef int i
    for i in range(2, n_samples):
        fs[i] = (14 * fs[i - 1] - 7 * fs[i - 2] + (samples[i] - samples[i - 2]) / 2) / 8
    
    cdef double fs_max, fs_mean, fs_md
    cdef np.ndarray[double, ndim=1] fs_segment
    for i in range(0, n_samples, sampling_rate):
        # these values are calculated for every 1-sec time interval
        fs_segment = fs[i:i + sampling_rate]
        fs_max = np.max(fs_segment)
        fs_mean = np.mean(fs_segment)
        fs_md = np.mean(np.abs(fs_segment - fs_mean))
        # 1) Count1—Range: 0.5∗max(|FS|) to max(|FS|);
        count1 += np.sum(fs_segment >= 0.5 * fs_max)

        # 2) Count2—Range: mean(|FS|) to max(|FS|);
        count2 += np.sum(fs_segment >= fs_mean)

        # 3) Count3—Range: mean(|FS|) – MD to mean(|FS|) + MD,
        count3 += np.sum(np.logical_and(fs_segment >= (fs_mean - fs_md), fs_segment <= (fs_mean + fs_md)))
    return (count1, count2, count3)


cdef emd_features(np.ndarray[double, ndim=1] samples, int sampling_rate, list imf_modes):
    imf_lz = array("d")
    # resample to 250 Hz
    if sampling_rate != 250:
        samples = signal.resample(samples, int((len(samples) / sampling_rate) * 250))
    # normalize to 0 - 1
    samples -= np.min(samples)
    samples /= np.max(samples)
    # normalize to 0 - 2^12 (convert to 12 bit resolution)
    cdef np.ndarray[np.uint16_t, ndim=1] emd_samples = (samples * (2 ** 12)).astype("uint16")
    # perform EMD
    imfs = emd(emd_samples, max_modes=max(imf_modes))
    cdef int imf_mode
    cdef np.ndarray[np.uint16_t, ndim=1] imf
    cdef bytearray bin_imf
    cdef bint little_endian
    for imf_mode in imf_modes:
        # IMF1_LZ: LZ complexity of EMD IMF
        imf = imfs[imf_mode - 1].astype("uint16")
        # determine byte_order
        little_endian = True if sys.byteorder == "little" else False
        bin_imf = array_to_binary_byte_string(array("H", imf), 12, little_endian)  # H means unsigned short int.
        imf_lz.append(lempel_ziv_complexity(bin_imf))
    return imf_lz


cdef tuple beat_statistics(list beats):
    cdef double rr_average = 0.0
    cdef double rr_std = 0.0
    cdef double unknown_beats = 0.0
    cdef double vpc_beats = 0.0
    rr_intervals = array("d")
    cdef int last_beat_time, beat_time
    cdef str last_beat_type, beat_type
    cdef i
    cdef double rr_interval
    for i in range(1, len(beats)):
        last_beat_time, last_beat_type = beats[i - 1]
        beat_time, beat_type = beats[i]
        rr_interval = <double>(beat_time - last_beat_time) / 200.0  # sampling rate: 200
        rr_intervals.append(rr_interval)
        if beat_type == "Q":
            unknown_beats += 1
        elif beat_type == "V":
            vpc_beats += 1

    if rr_intervals:
        rr_average = np.mean(rr_intervals)
        rr_std = np.std(rr_intervals)

    if len(beats):
        unknown_beats /= len(beats)
        vpc_beats /= len(beats)
    return rr_average, rr_std, unknown_beats, vpc_beats


# Find the max peak-to-peak amplitude in the samples
# The amplitudes of samples should be converted to "mV" prior to calling this function.
cpdef double get_amplitude(np.ndarray[double, ndim=1] samples, int sampling_rate, plot=None):
    cdef double max_amplitude = 0.0
    cdef int half_peak_width = int(np.round(0.05 * sampling_rate))
    cdef np.ndarray[np.int_t, ndim=1] peak_indices = signal.argrelmax(samples, order=half_peak_width)[0]
    cdef np.ndarray[np.int_t, ndim=1] valley_indices = signal.argrelmin(samples, order=half_peak_width)[0]
    peak_iter = iter(peak_indices)
    valley_iter = iter(valley_indices)

    cdef int next_peak_idx = next(peak_iter, -1)
    cdef int next_valley_idx = next(valley_iter, -1)
    cdef int peak_idx = next_peak_idx
    cdef int valley_idx = next_valley_idx
    cdef double peak_val, valley_val, amplitude
    while peak_idx != -1 and valley_idx != -1:
        peak_val = samples[peak_idx]
        valley_val = samples[valley_idx]
        if peak_idx < valley_idx:  # we are at a peak now. go to the next valley
            while next_peak_idx < next_valley_idx and next_peak_idx != -1:  # skip adjacent peaks
                next_peak_idx = next(peak_iter, -1)
                if next_peak_idx != -1 and samples[next_peak_idx] > peak_val:
                    peak = samples[next_peak_idx]
        else:  # we are at a valley now. go to the next peak
            while next_valley_idx < next_peak_idx and next_valley_idx != -1:  # skip adjacent valleys
                next_valley_idx = next(valley_iter, -1)
                if next_valley_idx != -1 and samples[next_valley_idx] < valley_val:
                    valley_val = samples[next_valley_idx]
        amplitude = abs(peak_val - valley_val)
        if amplitude > max_amplitude:
            max_amplitude = amplitude
        peak_idx = next_peak_idx
        valley_idx = next_valley_idx

    if plot:
        plot.plot(peak_indices, samples[peak_indices], color="b", marker="o", linestyle="")
        plot.plot(valley_indices, samples[valley_indices], color="g", marker="o", linestyle="")
    return max_amplitude


# extract features from raw sample points of the original ECG signal
# Before calling this function, the samples should have ADC zero substracted and converted to mV.
cpdef extract_features(np.ndarray[double, ndim=1] samples, int sampling_rate, set features_to_extract=feature_names_set):
    result = array("d")

    # Pre-processing:
    # Reference: https://homepages.fhv.at/ku/karl/VF/filtering.m

    # perform mean subtraction
    samples = samples - np.mean(samples)

    # 5-order moving average
    samples = moving_average(samples, order=5)

    # low pass filter for drift supression
    samples = drift_supression(samples, 1, sampling_rate)

    # band pass filter
    samples = butter_lowpass_filter(samples, 30, sampling_rate)

    cdef int n_samples = len(samples)

    # calculate the raw amplitude of the signals
    cdef double amplitude = get_amplitude(samples, sampling_rate)

    # perform QRS beat detection as part of the feature extraction process
    cdef list beats = qrs_detect(samples, sampling_rate)

    # We move the normalization step to the end of pre-processing since we need amplitudes of the original signals.
    # normalize the input ECG sequence
    samples = (samples - np.min(samples)) / (np.max(samples) - np.min(samples))

    # Time domain/morphology features
    # -------------------------------------------------
    # Threshold crossing interval (TCI) and Threshold crossing sample count (TCSC)

    # get all crossing points, use 20% of maximum as threshold
    # calculate average TCSC using a 3-s window
    # using 3-s moving window
    cdef double tcsc = 0.0
    if "TCSC" in features_to_extract:
        tcsc = threshold_crossing_sample_counts(samples, sampling_rate=sampling_rate, window_duration=3.0, threshold_ratio=0.2)
    result.append(tcsc)

    # average TCI for every 1-second segments
    cdef double tci = 0.0
    if "TCI" in features_to_extract:
        tci = threshold_crossing_intervals(samples, sampling_rate=sampling_rate, threshold_ratio=0.2)
    result.append(tci)

    # Standard exponential (STE)
    cdef double ste = 0.0
    if "STE" in features_to_extract:
        ste = standard_exponential(samples, sampling_rate)
    result.append(ste)

    # Modified exponential (MEA)
    cdef double mea = 0.0
    if "MEA" in features_to_extract:
        mea = modified_exponential(samples, sampling_rate)
    result.append(mea)

    # phase space reconstruction (PSR)
    cdef double psr = 0.0
    if "PSR" in features_to_extract:
         psr = phase_space_reconstruction(samples, sampling_rate)
    result.append(psr)  # phase space reconstruction (PSR)

    # Hilbert transformation + PSR
    cdef double hilb = 0.0
    if "HILB" in features_to_extract:
        hilb = hilbert_psr(samples, sampling_rate)
    result.append(hilb)

    cdef double vf = 0.0
    cdef double spectral_moment = 0.0, a2 = 0.0
    cdef double central_freq = 0.0

    cdef np.ndarray[double complex, ndim=1] fft
    cdef np.ndarray[double, ndim = 1] fft_freq
    cdef int n_fft
    if features_to_extract & {"VF", "M", "A2"}:
        # spectral parameters (characteristics of power spectrum)
        # -------------------------------------------------
        # perform discrete Fourier transform
        # apply a hamming window here for side lobe suppression.
        # (the original VF leak paper does not seem to do this).
        # http://www.ni.com/white-paper/4844/en/
        fft = np.fft.fft(samples * signal.hamming(n_samples))
        fft_freq = np.fft.fftfreq(n_samples)
        # We only need the left half of the FFT result (with frequency > 0)
        n_fft = np.ceil(n_samples / 2)
        fft = fft[0:n_fft]
        fft_freq = fft_freq[0:n_fft]

        # calculate VF leaks
        if "VF" in features_to_extract:
            vf = vf_leak(samples, fft, fft_freq)

        # calculate other spectral parameters (M and A2)
        if features_to_extract & {"M", "A2"}:
            (spectral_moment, a2) = spectral_features(fft, fft_freq, sampling_rate)
            if "M" not in features_to_extract:
                spectral_moment = 0.0
            elif "A2" not in features_to_extract:
                a2 = 0.0

        # central frequency (FM)
        if "FM" in features_to_extract:
            central_freq = central_frequency(fft, fft_freq, sampling_rate)

    result.append(vf)  # VF
    result.append(spectral_moment)  # M
    result.append(a2)  # A2
    result.append(central_freq)  # FM

    # complexity parameters
    # -------------------------------------------------
    cdef double lzc = 0.0
    if "LZ" in features_to_extract:
        lzc = complexity_measure(samples)
    result.append(lzc)

    # sample entropy (SpEn)
    cdef double spen = 0.0
    if "SpEn" in features_to_extract:
        spen = sample_entropy(samples)
    result.append(spen)

    # MAV
    cdef double mav = 0.0
    if "MAV" in features_to_extract:
        mav = mean_absolute_value(samples, sampling_rate)
    result.append(mav)

    # additional new features
    # -------------------------------------------------

    # Auxiliary counts (count1, 2, 3)
    cdef int count1 = 0, count2 = 0, count3 = 0
    if features_to_extract & {"Count1", "Count2", "Count3"}:
        (count1, count2, count3) = auxiliary_counts(samples, sampling_rate)
        if "Count1" not in features_to_extract:
            count1 = 0
        if "Count2" not in features_to_extract:
            count2 = 0
        if "Count3" not in features_to_extract:
            count3 = 0
    result.append(count1)
    result.append(count2)
    result.append(count3)

    # amplitude of the raw signals
    result.append(amplitude if "Amplitude" in features_to_extract else 0.0)

    cdef double imf1_lz = 0.0, imf5_lz = 0.0
    cdef bytearray bin_imf
    # Empirical mode decomposition (EMD)
    if features_to_extract & {"IMF1_LZ", "IMF5_LZ"}:
        # FIXME: allow only update one IMF mode
        imf_modes = [1, 5]
        imf_lz = emd_features(samples, sampling_rate, imf_modes)
        # IMF1_LZ, LZ complexity of EMD IMF1
        if "IMF1_LZ" in features_to_extract:
            imf1_lz = imf_lz[0]
        # IMF5_LZ, LZ complexity of EMD IMF5
        if "IMF5_LZ" in features_to_extract:
            imf5_lz = imf_lz[1]
    result.append(imf1_lz)
    result.append(imf5_lz)

    cdef double rr_avg, rr_std, unknwon_ratio, vpc_ratio, rr_cv
    (rr_avg, rr_std, unknwon_ratio, vpc_ratio) = beat_statistics(beats)
    rr_cv = rr_std / rr_avg if rr_avg else 0.0
    result.append(rr_avg if "RR" in features_to_extract else 0.0)
    result.append(rr_std if "RR_Std" in features_to_extract else 0.0)
    result.append(rr_cv if "RR_CV" in features_to_extract else 0.0)
    result.append(unknwon_ratio if "UR" in features_to_extract else 0.0)
    result.append(vpc_ratio if "VR" in features_to_extract else 0.0)

    return result, beats, amplitude


# load features from data file
def load_features(features_file):
    x_data = []
    x_data_info = []
    with open(features_file, "rb") as f:
        x_data = pickle.load(f)
        x_data_info = pickle.load(f)

    x_data = np.array(x_data)
    x_data_info = np.array(x_data_info)
    return x_data, x_data_info
