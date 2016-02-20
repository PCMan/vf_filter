#!/usr/bin/env python2
#  wfdb.py
#  
#  Copyright 2016 Unknown <pcman@arch-pc>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  

from ctypes import *
import numpy as np
import wfdb
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans

if __name__ == "__main__":
    record_name = "mitdb/108"

    s = (wfdb.WFDB_Siginfo  * 2)()
    v = (wfdb.WFDB_Sample * 2)()
    lead2 = []
    v5 = []
    if wfdb.isigopen(record_name, byref(s), 2) == 2:
        for i in range(1000):
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
            print ann.time, wfdb.anndesc(ann_code), wfdb.annstr(ann_code)
            anns.append((ann.time, ann_code))

    plt.plot(lead2)
    plt.plot(v5)

    first_beat = 0
    last_beat = -1
    beats = []
    for x, code in anns:
        if x >= len(lead2):
            break
        y = max(lead2[x], v5[x]) + 10
        plt.annotate(wfdb.annstr(code), xy=(x, y))
        if code == wfdb.NORMAL:
            if last_beat >= 0:
                beats.append(v5[last_beat:x])
            else:
                first_beat = x
            last_beat = x

    beat = beats[0]
    km = KMeans(20)
    x_data = [x for x in zip(range(len(beat)), beat)]
    km.fit(x_data)

    km.cluster_centers_.sort(axis=0)
    print km.cluster_centers_
    x = [first_beat + row[0] for row in km.cluster_centers_]
    y = [row[1] for row in km.cluster_centers_]
    plt.plot(x, y, "o")

    plt.show()
