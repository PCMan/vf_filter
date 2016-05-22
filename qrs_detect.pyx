import pyximport; pyximport.install()  # use Cython
import numpy as np
import scipy as sp
import scipy.signal
import datetime
import threading


# defined in osea20-gcc/bdac.c
cdef extern int BeatDetectAndClassify(int ecgSample, int *beatType, int *beatMatch)
cdef extern void ResetBDAC()

# the underlying C library used for QRS detection is not thread-safe nor reentrant.
# So, use a lock here to only allow calling from a thread at a time.
_lock = threading.Lock()

# This is not thread-safe due to the design flaw in osea20
def qrs_detect(signals, int samplingRate, double gain):
    beats = []
    # resample to 200Hz if needed
    if samplingRate != 200:
        signals = sp.signal.resample(signals, (len(signals) / samplingRate) * 200)

    _lock.acquire()
    ResetBDAC()  # reset the QRS detector
    cdef int beat_type = 0, beat_match = 0
    cdef double tmp
    cdef int sample
    for sample_count, sample in enumerate(signals):  # send the samples to the beat detector one by one
        # Set baseline to 0 and resolution to 5 mV/lsb (200 units/mV)
        # lTemp = ecg[0]-ADCZero ;
        tmp = sample
        tmp *= 200
        tmp /= gain
        sample = int(tmp)
        delay = BeatDetectAndClassify(sample, &beat_type, &beat_match)
        if delay:
            beat_time = sample_count - delay
            time_str = str(datetime.timedelta(seconds=(beat_time / samplingRate)))
            print(time_str, beat_type)
            beats.append(beat_time)
    _lock.release()
    return beats
