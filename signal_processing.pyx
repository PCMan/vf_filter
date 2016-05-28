#!/usr/bin/env python3
# This implements common VF related features reported in the literatures.
# Felipe AA et al. 2014. Detection of Life-Threatening Arrhythmias Using Feature Selection and Support Vector Machines
import pyximport; pyximport.install()  # use Cython
import numpy as np
cimport numpy as np  # use cython to speed up numpy array access (http://docs.cython.org/src/userguide/numpy_tutorial.html)
import scipy.signal as signal


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



cpdef double lempel_ziv_complexity(bytearray bin_str):
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


# Find the max peak-to-peak amplitude in the samples
# The amplitudes of samples should be converted to "mV" prior to calling this function.
cpdef double get_amplitude(np.ndarray[double, ndim=1] samples, int sampling_rate, plot=None):
    # 5-order moving average
    samples = moving_average(samples, order=5)

    # low pass filter for drift supression
    samples = drift_supression(samples, 1, sampling_rate)

    # band pass filter
    samples = butter_lowpass_filter(samples, 30, sampling_rate)

    # find all local maximum and minimum
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
