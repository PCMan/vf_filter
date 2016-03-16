#!/usr/bin/env python2

from ctypes import *
import numpy as np
import os
import wfdb
import matplotlib.pyplot as plt 


# get name of records from a database
def get_records(name):
    records = []
    dirname = os.path.expanduser("~/database/{0}".format(name))
    for name in os.listdir(dirname):
        if name.endswith(".dat"):
            record = os.path.splitext(name)[0]
            records.append(record)
    records.sort()
    return records


if __name__ == "__main__":
    # mitdb and vfdb contain two channels, but we only use the first one here
    for db_name in ("mitdb", "vfdb", "cudb"):
        for record in get_records(db_name):
            record_name = "{0}/{1}".format(db_name, record)
            print "READ SIGNAL:", record_name

            # query number of channels in this record
            n_channels = wfdb.isigopen(record_name, None, 0)

            # read the signals
            sigInfo = (wfdb.WFDB_Siginfo * n_channels)()
            sample_buf = (wfdb.WFDB_Sample * n_channels)()
            signals = []
            if wfdb.isigopen(record_name, byref(sigInfo), n_channels) == n_channels:
                while True:
                    if wfdb.getvec(byref(sample_buf)) < 0:
                        break
                    sample = sample_buf[0]  # we only want the first channel
                    signals.append(sample)
            signals = np.array(signals)
            print "  # of samples:", len(signals)
            
            # read annotations
            annotations = []
            ann_name = wfdb.String("atr")
            ann_info = (wfdb.WFDB_Anninfo * n_channels)()
            for item in ann_info:
                item.name = ann_name
                item.stat = wfdb.WFDB_READ
            if wfdb.annopen(record_name, byref(ann_info), n_channels) == 0:
                ann_buf = (wfdb.WFDB_Annotation * n_channels)()
                for i in range(30):
                    if wfdb.getann(0, byref(ann_buf)) < 0:
                        break
                    ann = ann_buf[0]  # we only want the first channel
                    ann_code = ord(ann.anntyp)
                    rhythm_type = None
                    if ann.aux:
                        # the first byte of aux is the length of the string
                        aux_ptr = cast(ann.aux, c_void_p).value + 1  # skip the first byte
                        rhythm_type = cast(aux_ptr, c_char_p).value
                    print ann.time, wfdb.anndesc(ann_code), wfdb.annstr(ann_code), rhythm_type
                    annotations.append((ann.time, ann_code, rhythm_type))

    wfdb.wfdbquit()
