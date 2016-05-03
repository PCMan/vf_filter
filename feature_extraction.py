#!/usr/bin/env python3
import pyximport; pyximport.install()
import vf_data
import vf_features
import numpy as np
from joblib import Parallel, delayed
from joblib.pool import has_shareable_memory
import scipy.signal as signal
import pickle
import argparse


def extract_features(idx, segment, sampling_rate, verbose):
    info = segment.info
    segment_duration = info.get_duration()
    signals = segment.signals
    # resample to DEFAULT_SAMPLING_RATE as needed
    if info.sampling_rate != sampling_rate:
        signals = signal.resample(signals, int(sampling_rate * segment_duration))

    features = vf_features.extract_features(signals, sampling_rate)
    print("{0}: {1}/{2}".format(idx, info.record_name, info.begin_time))
    if verbose:
        print("\t", features)
    return idx, features, info


# this is a generator function
def load_all_segments(segment_duration):
    idx = 0
    dataset = vf_data.DataSet()
    for segment in dataset.get_samples(segment_duration):
        yield idx, segment
        idx += 1


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("-s", "--segment-duration", type=int, default=8)
    parser.add_argument("-r", "--sampling-rate", type=int, default=250)
    parser.add_argument("-j", "--jobs", type=int, default=-1)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    x_data_info = []
    x_data = []
    # perform segmentation + feature extraction
    parellel = Parallel(n_jobs=args.jobs, verbose=0, backend="multiprocessing", max_nbytes=2048)
    results = parellel(delayed(extract_features)(idx, segment, args.sampling_rate, args.verbose) for idx, segment in load_all_segments(args.segment_duration))
    # sort the results from multiple jobs according to the order they are emitted
    results.sort(key=lambda result: result[0])
    for idx, features, segment_info in results:
        x_data_info.append(segment_info)
        x_data.append(features)

    # write to output files
    with open(args.output, "wb") as f:
        pickle.dump(x_data, f)
        pickle.dump(x_data_info, f)


if __name__ == "__main__":
    main()
