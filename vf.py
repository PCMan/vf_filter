#!/usr/bin/env python2

from ctypes import *
import numpy as np
import wfdb
import matplotlib.pyplot as plt 

if __name__ == "__main__":
    records = []
    with open("/home/pcman/database/vfdb/RECORDS", "r") as f:
        for line in f:
            records.append(line.strip())

    for record in records:
        record_name = "vfdb/{0}".format(record)

        s = (wfdb.WFDB_Siginfo  * 2)()
        v = (wfdb.WFDB_Sample * 2)()
        lead2 = []
        v5 = []
        if wfdb.isigopen(record_name, byref(s), 2) == 2:
            for i in range(10000):
                if wfdb.getvec(byref(v)) < 0:
                    break
                lead2.append(v[0])
                v5.append(v[1])
        lead2 = np.array(lead2)
        v5 = np.array(v5)

        anns= []
        ann_name = wfdb.String("atr")
        ann_info = (wfdb.WFDB_Anninfo * 2)()
        ann_info[0].name = ann_name
        ann_info[0].stat = wfdb.WFDB_READ
        ann_info[1].name = ann_name
        ann_info[1].stat = wfdb.WFDB_READ
        if wfdb.annopen(record_name, byref(ann_info), 2) == 0:
            ann = wfdb.WFDB_Annotation()
            for i in range(30):
                if wfdb.getann(0, byref(ann)) < 0:
                    break
                ann_code = ord(ann.anntyp)

                # the first byte of aux is the length of the string
                paux = cast(ann.aux, c_void_p).value + 1  # skip the first byte
                rhythm_type = cast(paux, c_char_p).value
                print ann.time, wfdb.anndesc(ann_code), wfdb.annstr(ann_code), rhythm_type

                anns.append((ann.time, ann_code))

        plt.plot(lead2)
        plt.plot(v5)
        for x, code in anns:
            if x >= len(lead2):
                break
            y = max(lead2[x], v5[x]) + 10
            plt.annotate(wfdb.annstr(code), xy=(x, y))
        plt.show()

    wfdb.wfdbquit()
