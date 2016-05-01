#!/usr/bin/env python3
from array import array
from libc.stdlib cimport malloc, free
import numpy as np

# use wfdb APIs
cdef extern from "<wfdb/wfdb.h>":
    ctypedef long WFDB_Time
    ctypedef int WFDB_Sample
    ctypedef double	WFDB_Frequency
    ctypedef char* FSTRING
    ctypedef WFDB_Frequency FFREQUENCY
    ctypedef int FINT
    ctypedef unsigned int WFDB_Annotator
    ctypedef double	WFDB_Gain;
    ctypedef unsigned int WFDB_Group

    ctypedef struct WFDB_Anninfo:
        char *name
        int stat

    ctypedef struct WFDB_Siginfo:
        char *fname
        char *desc
        char *units
        WFDB_Gain gain
        WFDB_Sample initval
        WFDB_Group group
        int fmt
        int spf
        int bsize
        int adcres
        int adczero
        int baseline
        long nsamp
        int cksum

    ctypedef struct WFDB_Annotation:
        WFDB_Time time
        char anntyp
        signed char subtyp
        unsigned char chan
        signed char num
        unsigned char *aux

    FINT annopen(char *record, WFDB_Anninfo *aiarray, unsigned int nann)
    FINT isigopen(char *record, WFDB_Siginfo *siarray, int nsig)
    FFREQUENCY sampfreq(char *record)
    FINT getvec(WFDB_Sample *vector)
    FINT getann(WFDB_Annotator a, WFDB_Annotation *annot);
    FSTRING annstr(int annotation_code)
    void wfdbquit()

    # constants
    int WFDB_DEFGAIN
    int WFDB_READ

# end of wfdb API declarations

def read_info(str record_name):
    _record_name = bytes(record_name, encoding="ascii")
    # query number of channels in this record
    cdef int n_channels = isigopen(_record_name, NULL, 0)
    # query sampling rate of the record
    cdef WFDB_Frequency sampling_rate = sampfreq(_record_name)
    return n_channels, sampling_rate

def read_signals(str record_name, int channel=0):
    _record_name = bytes(record_name, encoding="ascii")
    n_channels, sampling_rate = read_info(record_name)
    gain = WFDB_DEFGAIN
    # read the signals
    cdef WFDB_Siginfo* sigInfo = <WFDB_Siginfo*>malloc(n_channels * sizeof(WFDB_Siginfo))
    cdef WFDB_Sample* sample_buf = <WFDB_Sample*>malloc(n_channels * sizeof(WFDB_Sample))
    signals = array("i")  # signed int
    if isigopen(_record_name, sigInfo, n_channels) == n_channels:
        gain = sigInfo[0].gain
        while getvec(sample_buf) > 0:
            sample = sample_buf[channel]  # we only want the specified channel
            signals.append(sample)
    # free(sigInfo)
    # free(sample_buf)
    return n_channels, sampling_rate, gain, np.array(signals)


def read_annotations(str record_name, str ann_name="atr", int channel=0):
    # query number of channels in this record
    _record_name = bytes(record_name, encoding="ascii")
    cdef int n_channels = isigopen(_record_name, NULL, 0)
    # read annotations
    annotations = []
    cdef WFDB_Anninfo* ann_info = <WFDB_Anninfo*>malloc(n_channels * sizeof(WFDB_Anninfo))
    cdef WFDB_Annotation* ann_buf
    cdef i
    _ann_name = bytes(ann_name, encoding="ascii")
    for i in range(n_channels):
        ann_info[i].name = _ann_name
        ann_info[i].stat = WFDB_READ
    cdef WFDB_Annotation* ann
    if annopen(_record_name, ann_info, n_channels) == 0:
        ann_buf = <WFDB_Annotation*>malloc(n_channels * sizeof(WFDB_Annotation))
        while getann(0, ann_buf) == 0:
            ann = &ann_buf[channel]  # we only want the specified channel
            time = ann.time
            code = str(annstr(<int>ann.anntyp), encoding="ascii")
            sub_type = <int>ann.subtyp
            rhythm_type = ""
            if ann.aux != NULL:
                # the first byte of aux is the length of the string
                rhythm_type = str(<char*>ann.aux + 1, encoding="ascii")
            # print(time, code, sub_type, rhythm_type)
            annotations.append((time, code, sub_type, rhythm_type))
        free(ann_buf)
    free(ann_info)
    return annotations


def quit():
    wfdbquit()
