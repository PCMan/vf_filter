#!/usr/bin/env python3
import pyximport; pyximport.install()
import vf_features
import vf_data
import argparse
import numpy as np
import matplotlib.pyplot as plt
import signal_processing
from scipy import signal


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

    # trend removal (drift suppression)
    samples = signal_processing.drift_supression(samples, 1, record.sampling_rate)

    # smoothing
    samples = signal_processing.moving_average(samples)

    # plot the signals
    fig, axes = plt.subplots(4)

    ax = axes[0]
    ax.hlines(np.arange(-2, 3, step=0.5), xmin=0, xmax=len(samples), color="r")
    ax.plot(samples, color="k")

    amplitude = vf_features.get_amplitude(samples, record.sampling_rate, plot=ax)
    print("amplitude:", amplitude)

    ax = axes[1]
    n_samples = len(samples)
    fft = np.fft.fft(samples * signal.hamming(n_samples))
    fft_freq = np.fft.fftfreq(n_samples)
    # We only need the left half of the FFT result (with frequency > 0)
    n_fft = int(np.ceil(n_samples / 2))
    fft = fft[0:n_fft]
    fft_freq = fft_freq[0:n_fft]
    ax.plot(fft_freq, np.abs(fft))

    ax = axes[2]
    ax.set_title("After preprocessing")
    ax.plot(samples, color="k")
    """
    # log signal
    ax = axes[2]
    log_samples = np.log(samples - np.min(samples) + 0.1)# + np.finfo("float64").eps)
    ax.plot(log_samples, color="k")
    ax.plot(samples, color="k")
    
    ax = axes[3]
    n_samples = len(log_samples)
    fft = np.fft.fft(log_samples * signal.hamming(n_samples))
    fft_freq = np.fft.fftfreq(n_samples)
    # We only need the left half of the FFT result (with frequency > 0)
    n_fft = int(np.ceil(n_samples / 2))
    fft = fft[0:n_fft]
    fft_freq = fft_freq[0:n_fft]
    ax.plot(fft_freq, np.abs(fft))
    """

    plt.show()


if __name__ == '__main__':
    main()
