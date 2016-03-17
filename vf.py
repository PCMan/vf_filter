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


class Record:
    def __init__(self):
        self.signals = []
        self.annotations = []
        self.name = ""
        self.sample_rate = 0

    def load(self, db_name, record):
        record_name = "{0}/{1}".format(db_name, record)
        self.name = record_name

        # query number of channels in this record
        n_channels = wfdb.isigopen(record_name, None, 0)

        # query sampling rate of the record
        self.sample_rate = wfdb.sampfreq(record_name)

        # read the signals
        sigInfo = (wfdb.WFDB_Siginfo * n_channels)()
        sample_buf = (wfdb.WFDB_Sample * n_channels)()
        signals = []
        if wfdb.isigopen(record_name, byref(sigInfo), n_channels) == n_channels:
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

        self.signals = signals
        self.annotations = annotations

    def find_vf_episodes(self):
        annotations = self.annotations
        n_samples = len(self.signals)
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
            episodes.append((vf_begin, n_samples))

        return episodes


if __name__ == "__main__":
    # mitdb and vfdb contain two channels, but we only use the first one here
    # data source sampling rate:
    # mitdb: 360 Hz
    # vfdb, cudb: 250 Hz
    for db_name in ("mitdb", "vfdb", "cudb"):
        for record_name in get_records(db_name):
            print "READ SIGNAL:", db_name, record_name
            record = Record()
            record.load(db_name, record_name)

            # resample to 250 Hz for all signals
            print "sample rate:", record.sample_rate, "# of samples:", len(record.signals), ", # of anns:", len(record.annotations)
            print "vf episodes:", record.find_vf_episodes()

    wfdb.wfdbquit()
