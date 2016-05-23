# This is a quick and dirty python wrapper for the free QRS detector developed by EP Limited
# PS Hamilton, WJ Tompkins. Quantitative investigation of QRS detection rules using the MIT/BIH arrhythmia database.
# IEEE Trans. Biomed. Eng BME-33: 1158-1165 (1987).
# http://www.eplimited.com/software.htm
# The C code is put in osea20-gcc directory and it's licensed under GNU Library General Public License (LGPL).

import pyximport; pyximport.install()  # use Cython
import numpy as np
cimport numpy as np  # use cython to speed up numpy array access (http://docs.cython.org/src/userguide/numpy_tutorial.html)
import scipy as sp
import scipy.signal
import threading


# defined in osea20-gcc/bdac.c
cdef extern int BeatDetectAndClassify(int ecgSample, int *beatType, int *beatMatch)
cdef extern void ResetBDAC()

# defined in osea20-gcc/bxbep.c
cdef extern int amap(int a)

# the underlying C library used for QRS detection is not thread-safe nor reentrant.
# So, use a lock here to only allow calling from a thread at a time.
_lock = threading.Lock()

# This is not thread-safe due to the design flaw in osea20
# Before calling this function, the signals should be converted to mV and with ADC zero subtracted
cpdef list qrs_detect(np.ndarray[double, ndim=1] src_signals, int sampling_rate):
    cdef list beats = []
    # resample to 200Hz if needed
    if sampling_rate != 200:
        src_signals = sp.signal.resample(src_signals, (len(src_signals) / sampling_rate) * 200)

    # Set baseline to 0 and resolution to 5 mV/lsb (200 units/mV)
    cdef np.ndarray[np.int_t, ndim=1] signals = (src_signals * 200).astype("int")

    _lock.acquire()
    ResetBDAC()  # reset the QRS detector
    cdef int beat_type = 0, beat_match = 0
    cdef int sample
    # The QRS detector need to wait for 8 QRS beats to initialize the thresholds.
    # Let's feed the  samples to initialize it and let it reach a steady state.
    cdef int n_beats = 0
    for sample in signals:
        beat_type = 0
        beat_match = 0
        if BeatDetectAndClassify(sample, &beat_type, &beat_match):
            n_beats += 1
            if n_beats >= 8:
                break

    # now the beat detector already read len(signals) samples and is possibly in a steady state.
    # Let's feed the same samples to it again to do the real detection.
    for sample_count, sample in enumerate(signals):  # send the samples to the beat detector one by one
        beat_type = 0
        beat_match = 0
        delay = BeatDetectAndClassify(sample, &beat_type, &beat_match)
        if delay:
            beat_time = sample_count - delay
            if beat_time < 0:
                continue
            type_code = amap(beat_type)
            beats.append((beat_time, chr(type_code)))
    _lock.release()
    return beats
