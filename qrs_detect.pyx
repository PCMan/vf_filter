# This is a quick and dirty python wrapper for the free QRS detector developed by EP Limited
# PS Hamilton, WJ Tompkins. Quantitative investigation of QRS detection rules using the MIT/BIH arrhythmia database.
# IEEE Trans. Biomed. Eng BME-33: 1158-1165 (1987).
# http://www.eplimited.com/software.htm
# The C code is put in osea20-gcc directory and it's licensed under GNU Library General Public License (LGPL).

import pyximport; pyximport.install()  # use Cython
import numpy as np
import scipy as sp
import scipy.signal
import datetime
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
def qrs_detect(signals, int sampling_rate, double gain):
    beats = []
    # resample to 200Hz if needed
    if sampling_rate != 200:
        signals = sp.signal.resample(signals, (len(signals) / sampling_rate) * 200)

    _lock.acquire()
    ResetBDAC()  # reset the QRS detector
    cdef int beat_type = 0, beat_match = 0
    cdef double tmp
    cdef int sample
    for sample_count, sample in enumerate(signals):  # send the samples to the beat detector one by one
        # Set baseline to 0 and resolution to 5 mV/lsb (200 units/mV)
        # FIXME: remove ADC zero
        # tmp = sample-ADCZero ;
        sample = <int>(sample * 200 / gain)
        beat_type = 0
        beat_match = 0
        delay = BeatDetectAndClassify(sample, &beat_type, &beat_match)
        if delay:
            beat_time = sample_count - delay
            # time_str = str(datetime.timedelta(seconds=(beat_time / sampling_rate)))
            type_code = amap(beat_type)
            # print(time_str, chr(type_code))
            beats.append((beat_time, chr(type_code)))
    _lock.release()
    return beats
