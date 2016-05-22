#!/usr/bin/env python3
import pyximport; pyximport.install()  # use Cython
import qrs_detect
import vf_data
import matplotlib.pyplot as plt
import scipy as sp
import scipy.signal


def main(args):
    record = vf_data.Record()
    record.load("mitdb", "100", 0, "atr")
    nsig = int(30 * record.sampling_rate)
    signals = record.signals[:nsig]
    # resmaple to 200
    signals = sp.signal.resample(signals, (len(signals) / record.sampling_rate) * 200)
    beats = qrs_detect.qrs_detect(signals, 200, record.gain)
    plt.plot(signals)
    plt.vlines(beats, 0, max(signals))
    plt.show()
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
