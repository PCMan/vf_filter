#!/usr/bin/env python3
import pyximport; pyximport.install()
import vf_features
import vf_data
import argparse
import scipy as sp
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt


def main():
    # parse command line arguments
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

    record.load(db_name, record_name, channel=channel, annotator=annotator)
    samples = record.signals[args.begin:(args.begin + args.duration * int(record.sampling_rate))]

    # convert to mV
    samples = (samples.astype("float64") - record.adc_zero) / record.gain

    # trend removal (drift suppression)
    samples = vf_features.drift_supression(samples, 1, record.sampling_rate)

    # smoothing
    samples = vf_features.moving_average(samples)

    # plot the signals
    fig, axes = plt.subplots(2)

    ax = axes[0]
    ax.hlines(np.arange(-2, 3, step=0.5), xmin=0, xmax=len(samples), color="r")
    ax.plot(samples, color="k")

    amplitude = vf_features.get_amplitude(samples, record.sampling_rate, plot=ax)
    print("amplitude:", amplitude)

    plt.show()


if __name__ == '__main__':
    main()
