#!/usr/bin/env python2
# convet binary data in wfdb formats to python pickle protocol2
import cPickle as pickle
import sys
import os
from ctypes import *
import wfdb
from array import array


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


def main():
    data_dir = "datasets"
    for db_name in ("mitdb", "vfdb", "cudb", "edb"):
        db_dir = os.path.join(data_dir, db_name)
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)

        for record in get_records(db_name):
            path = os.path.join(db_dir, record)
            if os.path.exists(path):  # skip files that are already converted
                continue

            print "read record:", db_name, record
            record_name = "{0}/{1}".format(db_name, record)

            # query number of channels in this record
            n_channels = wfdb.isigopen(record_name, None, 0)

            # query sampling rate of the record
            sampling_rate = wfdb.sampfreq(record_name)
            gain = wfdb.WFDB_DEFGAIN

            # read the signals
            sigInfo = (wfdb.WFDB_Siginfo * n_channels)()
            sample_buf = (wfdb.WFDB_Sample * n_channels)()
            signals = array("i")  # signed int
            if wfdb.isigopen(record_name, byref(sigInfo), n_channels) == n_channels:
                gain = sigInfo[0].gain
                while wfdb.getvec(byref(sample_buf)) > 0:
                    sample = sample_buf[0]  # we only want the first channel
                    signals.append(sample)

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

                    time = ann.time
                    ann_code = ord(ann.anntyp)
                    code = str(wfdb.annstr(ann_code))
                    sub_type = ord(ann.subtyp)
                    rhythm_type = ""
                    if ann.aux:
                        # the first byte of aux is the length of the string
                        aux_ptr = cast(ann.aux, c_void_p).value + 1  # skip the first byte
                        rhythm_type = cast(aux_ptr, c_char_p).value
                    annotations.append((time, code, sub_type, rhythm_type))

            with open(path, "wb") as f:
                pickle.dump((sampling_rate, gain), f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(signals, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(annotations, f, pickle.HIGHEST_PROTOCOL)

        wfdb.wfdbquit()

if __name__ == "__main__":
    main()
