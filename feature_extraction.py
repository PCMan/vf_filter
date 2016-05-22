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
import sys


def extract_features(idx, segment, resample_rate, update_features, verbose):
    info = segment.info
    segment_duration = info.get_duration()
    signals = segment.signals
    # resample to DEFAULT_SAMPLING_RATE as needed
    if resample_rate and info.resample_rate != resample_rate:
        signals = signal.resample(signals, int(resample_rate * segment_duration))
        sampling_rate = resample_rate
    else:
        sampling_rate = info.sampling_rate

    if update_features:
        feature_names = set(update_features)
    else:
        feature_names = vf_features.feature_names_set
    features = vf_features.extract_features(signals, sampling_rate, feature_names)
    print("{0}: {1}/{2}".format(idx, info.record_name, info.begin_time))
    if verbose:
        print("\t", features)
    return idx, features, info


# this is a generator function
def load_all_segments(args):
    idx = 0
    dataset = vf_data.DataSet()
    if args.correction_file:
        dataset.load_correction(args.correction_file)
    for segment in dataset.get_samples(args.segment_duration):
        yield idx, segment
        idx += 1


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("-s", "--segment-duration", type=int, default=8)
    parser.add_argument("-r", "--resample-rate", type=int, default=None)
    parser.add_argument("-j", "--jobs", type=int, default=-1)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-u", "--update-features", type=str, nargs="+", choices=vf_features.feature_names, default=None)
    parser.add_argument("-c", "--correction-file", type=str, help="Override the incorrect labels of the original dataset.")
    args = parser.parse_args()

    x_data_info = []
    x_data = []
    # perform segmentation + feature extraction
    parellel = Parallel(n_jobs=args.jobs, verbose=0, backend="multiprocessing", max_nbytes=2048)
    results = parellel(delayed(extract_features)(idx, segment, args.resample_rate, args.update_features, args.verbose) for idx, segment in load_all_segments(args))
    # sort the results from multiple jobs according to the order they are emitted
    results.sort(key=lambda result: result[0])
    for idx, features, segment_info in results:
        x_data_info.append(segment_info)
        x_data.append(features)

    # try to update existing feature file
    if args.update_features:
        # load the old file
        try:
            with open(args.output, "rb") as f:
                old_x_data = pickle.load(f)
                old_x_data_info = pickle.load(f)
                assert len(old_x_data) == len(x_data)
                # preserve features in the old file if it's not updated
                preserve_idx = [i for (i, name) in enumerate(vf_features.feature_names) if name not in args.update_features]
                # copy values we want to preserve from the old data to the new ones
                for new_x, new_info, old_x, old_info in zip(x_data, x_data_info, old_x_data, old_x_data_info):
                    assert (new_info.record_name == old_info.record_name and new_info.begin_time == old_info.begin_time)
                    for i in preserve_idx:
                        new_x[i] = old_x[i]
        except Exception:
            print("Unable to load", args.output, sys.exc_info())

    # write to output files
    with open(args.output, "wb") as f:
        pickle.dump(x_data, f)
        pickle.dump(x_data_info, f)


if __name__ == "__main__":
    main()
