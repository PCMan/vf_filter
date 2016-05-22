#!/usr/bin/env python3
from array import array
from libc.stdlib cimport malloc, free
import numpy as np
from wfdb cimport *


wfdbquiet()  # disable wfdb error output


def read_info(str record_name):
    _record_name = bytes(record_name, encoding="ascii")
    # query number of channels in this record_name
    cdef int n_channels = isigopen(_record_name, NULL, 0)
    # query sampling rate of the record_name
    cdef WFDB_Frequency sampling_rate = sampfreq(_record_name)
    return n_channels, sampling_rate

def read_signals(str record_name, int channel=0):
    _record_name = bytes(record_name, encoding="ascii")
    n_channels, sampling_rate = read_info(record_name)
    cdef WFDB_Gain gain = WFDB_DEFGAIN
    cdef int adc_zero = 0
    # read the signals
    cdef WFDB_Siginfo* sigInfo = <WFDB_Siginfo*>malloc(n_channels * sizeof(WFDB_Siginfo))
    cdef WFDB_Sample* sample_buf = <WFDB_Sample*>malloc(n_channels * sizeof(WFDB_Sample))
    signals = array("i")  # signed int
    if isigopen(_record_name, sigInfo, n_channels) == n_channels:
        gain = sigInfo[0].gain
        adc_zero = sigInfo[0].adczero
        while getvec(sample_buf) > 0:
            sample = sample_buf[channel]  # we only want the specified channel
            signals.append(sample)
    # free(sigInfo)
    # free(sample_buf)
    return n_channels, sampling_rate, gain, adc_zero, np.array(signals)


def read_annotations(str record_name, str ann_name="atr"):
    # query number of channels in this record_name
    _record_name = bytes(record_name, encoding="ascii")
    cdef int n_channels = isigopen(_record_name, NULL, 0)
    # read annotations
    annotations = []

    _ann_name = bytes(ann_name, encoding="ascii")
    cdef WFDB_Anninfo ann_info
    cdef WFDB_Annotation ann
    cdef i

    ann_info.name = _ann_name
    ann_info.stat = WFDB_READ
    if annopen(_record_name, &ann_info, 1) == 0:
        while getann(0, &ann) == 0:
            time = ann.time
            code = str(annstr(<int>ann.anntyp), encoding="ascii")
            sub_type = <int>ann.subtyp
            rhythm_type = ""
            if ann.aux != NULL:
                # the first byte of aux is the length of the string
                rhythm_type = str(<char*>ann.aux + 1, encoding="ascii")
            # print(time, code, sub_type, rhythm_type)
            annotations.append((time, code, sub_type, rhythm_type))
    return annotations


def quit():
    wfdbquit()
