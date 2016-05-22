#!/usr/bin/env python3
import pyximport; pyximport.install()  # use Cython
import qrs_detect
import vf_data
import matplotlib.pyplot as plt
import scipy as sp
import scipy.signal


def main(args):
    record = vf_data.Record()
    record.load("vfdb", "422", 0, "atr")
    n_sig = int(30 * record.sampling_rate)
    begin_sig = 385788
    signals = record.signals[begin_sig:begin_sig + n_sig]
    # resmaple to 200
    signals = sp.signal.resample(signals, (len(signals) / record.sampling_rate) * 200)
    beats = qrs_detect.qrs_detect(signals, 200, record.gain)
    print(beats)
    plt.plot(signals)
    plt.vlines([b[0] for b in beats], 0, max(signals))
    plt.show()
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
