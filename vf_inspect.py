#!/usr/bin/env python3
import pyximport; pyximport.install()
import vf_features
import vf_data
import argparse
import csv
from collections import deque
from datetime import timedelta
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt


MAX_QUEUE_SIZE = 50

record_cache = deque()  # a simple LRU cache for loaded records


def load_record(record_name):
    # find in the cache first (linear search is inefficient, but it works)
    record = None
    for i, cached_record in enumerate(record_cache):
        if record_name == cached_record.name:
            record = cached_record
            # move the record to end of the queue
            del record_cache[i]
            record_cache.append(record)
            break
    # not found in cache, load it
    if not record:
        record = vf_data.Record()
        names = record_name.split("/", maxsplit=1)
        db_name = names[0]
        record_id = names[1]
        channel = 1 if db_name == "mghdb" else 0
        annotator = "ari" if db_name == "mghdb" else "atr"
        record.load(db_name, record_id, channel, annotator)
        # insert into the cache
        record_cache.append(record)

        if len(record_cache) > MAX_QUEUE_SIZE:
            record_cache.popleft()  # remove one item from the cache
    return record


def plot_sample(info, signals):
    n_samples = len(signals)
    figure, axes = plt.subplots(3, 2)  # sharex=True
    ax = axes[0, 0]
    ax.set_title("before preprocessing")
    ax.plot(signals)

    # plot DFT spectrum
    fft = np.fft.fft(signals * sp.signal.hamming(n_samples))
    fft_freq = np.fft.fftfreq(n_samples)
    # We only need the left half of the FFT result (with frequency > 0)
    n_fft = int(np.ceil(n_samples / 2))
    fft = fft[0:n_fft]
    fft_freq = fft_freq[0:n_fft]
    amplitude = np.abs(fft)
    fft_freq_hz = fft_freq * info.sampling_rate
    ax = axes[0, 1]
    ax.set_title("DFT (before preprocessing)")
    ax.plot(fft_freq_hz, amplitude)

    # normalize the input ECG sequence
    signals = signals.astype("float64") # convert to float
    signals = (signals - np.min(signals)) / (np.max(signals) - np.min(signals))

    # perform mean subtraction
    signals = signals - np.mean(signals)

    # 5-order moving average
    signals = vf_features.moving_average(signals, order=5)
    ax = axes[1, 0]
    ax.set_title("moving average")
    ax.plot(signals)

    # band pass filter
    signals = vf_features.butter_bandpass_filter(signals, 0.5, 30, info.sampling_rate)
    ax = axes[2, 0]
    ax.set_title("band pass filter")
    ax.plot(signals)
    ax.axhline(y=0.2, color="r")  # draw a horizontal line at 0.2

    # plot DFT spectrum
    fft = np.fft.fft(signals * sp.signal.hamming(n_samples))
    fft_freq = np.fft.fftfreq(n_samples)
    # We only need the left half of the FFT result (with frequency > 0)
    n_fft = int(np.ceil(n_samples / 2))
    fft = fft[0:n_fft]
    fft_freq = fft_freq[0:n_fft]
    amplitude = np.abs(fft)
    fft_freq_hz = fft_freq * info.sampling_rate

    ax = axes[1, 1]
    ax.set_title("DFT (after preprocessing)")
    ax.plot(fft_freq_hz, amplitude)
    peak_freq_idx = np.argmax(amplitude)
    peak_freq = fft_freq[peak_freq_idx] * info.sampling_rate
    ax.axvline(x=peak_freq, color="r")

    # plot spectrogram for SPEC, M, and A2
    vf_features.spectral_features(fft, fft_freq, info.sampling_rate, plot=axes[2, 1])

    # maximize the window
    # Reference: http://stackoverflow.com/questions/12439588/how-to-maximize-a-plt-show-window-using-python
    fm = plt.get_current_fig_manager()
    if hasattr(fm, "window") and hasattr(fm.window, "showMaximized"):
        fm.window.showMaximized()
    plt.show()


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-f", "--features", type=str, required=True)
    parser.add_argument("-p", "--plot", action="store_true", default=False)
    parser.add_argument("-t", "--threshold", type=float, default=0.25)
    parser.add_argument("-r", "--rhythms", type=str, nargs="+")
    parser.add_argument("-d", "--db-names", type=str, nargs="+")
    args = parser.parse_args()

    # load features and info of the samples
    x_data, x_data_info = vf_features.load_features(args.features)

    fields = ["sample", "record", "begin", "rhythm", "tested", "errors", "error rate"]
    errors = []
    with open(args.input, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # we only want to inspect these types of rhythms
            if args.rhythms and row["rhythm"] not in args.rhythms:
                continue

            # we only want to inspect records of these databases
            if args.db_names:
                allowed = False
                for db_name in args.db_names:
                    if row["record"].startswith(db_name):
                        allowed = True
                        break
                if not allowed:
                    continue

            try:
                error_rate = float(row["error rate"])
                if error_rate > args.threshold:
                    errors.append({field: row[field] for field in fields})
            except ValueError:
                pass

    # sort the errors by error rate in descending order
    errors.sort(key=lambda item: item["error rate"], reverse=True)

    # output the errors
    print("{0} samples have error rate > {1}".format(len(errors) , args.threshold))
    for error in errors:
        print(", ".join([error[field] for field in fields]))
    print("-" * 80, "\n")
    # TODO: perform statistics for each rhythm type and record?

    fields.append("time")
    for error in errors:
        sample_idx = int(error["sample"]) - 1
        x_features = x_data[sample_idx]
        info = x_data_info[sample_idx]

        # convert sample count to time
        sample_time = timedelta(seconds=(info.begin_time / info.sampling_rate))
        error["time"] = sample_time

        print(", ".join(["{0}: {1}".format(field, error[field]) for field in fields]))

        # list each feature
        comments = {
            "TCSC": "(VF: > 48 for high Sp, 25-35 for high Se)",
            "TCI": "(SR: > 400)",
            "HILB": "(VF: > 0.15)",
            "PSR": "(VF: > 0.15)",
            "VF": "(VF: < 26/64 = {0}".format(26 / 64),
            "M": "(VF: <= 1.55)",
            "A2": "(VF: >= 0.45)",
            "LZ": "(SR: < 0.15, VT: 0.15 - 0.486, VF: > 0.486)",
            "SpEn": "(VF: > 0.25)"
        }
        for name, feature in zip(vf_features.feature_names, x_features):
            print("{0}: {1}\t{2}".format(name, feature, comments.get(name, "")))
        print("-" * 80)

        # plot the signals
        if args.plot:
            record = load_record(info.record_name)
            signals = record.signals[info.begin_time:info.end_time]

            # perform resample to 250 Hz
            if info.sampling_rate != 250:
                signals = sp.signal.resample(signals, int(info.get_duration() * 250))
                info.sampling_rate = 250
            plot_sample(info, signals)


if __name__ == '__main__':
    main()
