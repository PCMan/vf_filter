#!/usr/bin/env python3
import pyximport; pyximport.install()  # use Cython
import qrs_detect
import vf_data
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.signal
import datetime
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--record", type=str, required=True)
    parser.add_argument("-c", "--channel", type=int)
    parser.add_argument("-a", "--annotator", type=str)
    parser.add_argument("-b", "--begin", type=int, default=0)
    parser.add_argument("-d", "--duration", type=int, default=8)
    args = parser.parse_args()

    db_name, record_name = args.record.split("/", maxsplit=1)
    record = vf_data.Record()
    if args.channel is not None:
        channel = args.channel
    else:
        channel = 0 if db_name != "mghdb" else 1  # lead II is at channel 1 in mghdb

    if args.annotator is not None:
        annotator = args.annotator
    else:
        annotator = "atr" if db_name != "mghdb" else "ari"  # mghdb only contains ari annotations

    # record.load("vfdb", "422", 0, "atr")
    # begin_sig = 385788

    record.load(db_name, record_name, channel=channel, annotator=annotator)

    begin_sig = args.begin
    n_sig = int(args.duration * record.sampling_rate)
    signals = record.signals[begin_sig:begin_sig + n_sig]

    # resmaple to 200
    signals = sp.signal.resample(signals, (len(signals) / record.sampling_rate) * 200)
    beats = qrs_detect.qrs_detect(signals, sampling_rate=200)
    for beat_sample, beat_type in beats:
        time_str = str(datetime.timedelta(seconds=(beat_sample / 200)))
        print(time_str, beat_sample, beat_type)

    hr = (len(beats) / args.duration) * 60
    print("estimated HR:", hr, "BPM")

    plt.hlines(np.arange(-2, 3, step=0.5), xmin=0, xmax=len(signals), color="r")
    plt.plot(signals, color="k")
    plt.vlines([b[0] for b in beats], min(signals) - 1, max(signals) + 1, color="g")
    plt.show()
    return 0

if __name__ == '__main__':
    main()
