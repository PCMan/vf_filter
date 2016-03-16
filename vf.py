#!/usr/bin/env python2

from ctypes import *
import numpy as np
import os
import wfdb
import matplotlib.pyplot as plt 


# get name of records from a database
def get_records(db_name):
    records = []
    dirname = os.path.expanduser("~/database/{0}".format(db_name))
    for name in os.listdir(dirname):
        if name.endswith(".dat"):
            record = os.path.splitext(name)[0]
            records.append(record)
    records.sort()
    return records


def load_record(db_name, record, sample_rate=None):
    record_name = "{0}/{1}".format(db_name, record)

    # query number of channels in this record
    n_channels = wfdb.isigopen(record_name, None, 0)

    # read the signals
    sigInfo = (wfdb.WFDB_Siginfo * n_channels)()
    sample_buf = (wfdb.WFDB_Sample * n_channels)()
    signals = []
    if wfdb.isigopen(record_name, byref(sigInfo), n_channels) == n_channels:
        # ask wfdb to perform re-sampling
        if sample_rate:
            # https://www.physionet.org/physiotools/wpg/wpg_20.htm#getvec
            # If setifreq has been used to modify the input sampling rate, getvec resamples the input signals at the
            # desired rate, using linear interpolation between the pair of samples nearest in time to that of the
            # sample to be returned.
            wfdb.setifreq(wfdb.WFDB_Frequency(sample_rate))

        while wfdb.getvec(byref(sample_buf)) > 0:
            sample = sample_buf[0]  # we only want the first channel
            signals.append(sample)
    signals = np.array(signals)

    # read annotations
    annotations = []
    ann_name = wfdb.String("atr")
    ann_info = (wfdb.WFDB_Anninfo * n_channels)()
    for item in ann_info:
        item.name = ann_name
        item.stat = wfdb.WFDB_READ
    if wfdb.annopen(record_name, byref(ann_info), n_channels) == 0:
        ann_buf = (wfdb.WFDB_Annotation * n_channels)()
        while wfdb.getann(0, byref(ann_buf)) == 0:
            ann = ann_buf[0]  # we only want the first channel
            ann_code = ord(ann.anntyp)
            rhythm_type = ""
            if ann.aux:
                # the first byte of aux is the length of the string
                aux_ptr = cast(ann.aux, c_void_p).value + 1  # skip the first byte
                rhythm_type = cast(aux_ptr, c_char_p).value
            # print ann.time, wfdb.anndesc(ann_code), wfdb.annstr(ann_code), rhythm_type
            annotations.append((ann.time, wfdb.annstr(ann_code), rhythm_type))

    return signals, annotations


def find_vf_episodes(annotations):
    episodes = []
    vf_begin = -1
    for (ann_time, code, rhythm_type) in annotations:
        if vf_begin > 0:  # current rhythm is Vf
            if code == "]" or not rhythm_type.startswith("(V"):
                vf_end = ann_time
                episodes.append((vf_begin, vf_end))
                vf_begin = -1
        else:  # no Vf
            if code == "[" or rhythm_type.startswith("(V"):
                vf_begin = ann_time
    if vf_begin > 0:
        vf_end = annotations[-1][0]
        episodes.append((vf_begin, vf_end))

    return episodes


if __name__ == "__main__":
    # mitdb and vfdb contain two channels, but we only use the first one here
    # data source sampling rate:
    # mitdb: 360 Hz
    # vfdb, cudb: 250 Hz
    for db_name in ("mitdb", "vfdb", "cudb"):
        for record in get_records(db_name):
            print "READ SIGNAL:", db_name, record
            # resample to 250 Hz for all signals
            signals, annotations = load_record(db_name, record, sample_rate=250.0)
            print "  # of samples:", len(signals), ", # of anns:", len(annotations)
            print "vf episodes:", find_vf_episodes(annotations)

    wfdb.wfdbquit()
