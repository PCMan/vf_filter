#!/usr/bin/env python3
import pyximport; pyximport.install()
import vf_data
import vf_features
from array import array
import numpy as np
from joblib import Parallel, delayed
from joblib.pool import has_shareable_memory
import scipy.signal as signal
import pickle
import argparse


# label the segment according to different problem definition
def label_segment(segment_info):
    if segment_info.terminating_rhythm == vf_data.RHYTHM_VF or segment_info.terminating_rhythm == vf_data.RHYTHM_VFL:
        return 1
    return 0


def extract_features(idx, segment, sampling_rate):
    segment_duration = 8  # 8 sec per segment
    signals = segment.signals
    info = segment.info
    # resample to DEFAULT_SAMPLING_RATE as needed
    if info.sampling_rate != sampling_rate:
        signals = signal.resample(signals, sampling_rate * segment_duration)

    features = vf_features.extract_features(signals, sampling_rate)
    label = label_segment(info)
    print(info.record, info.begin_time, features, label)
    return idx, features, label, info


# this is a generator function
def load_all_segments(db_names, segment_duration):
    idx = 0
    for db_name in db_names:
        for record_name in vf_data.get_records(db_name):
            # load the record from the ECG database
            record = vf_data.Record()
            record.load(db_name, record_name)
            for segment in record.get_segments(segment_duration):
                yield idx, segment
                idx += 1


def main():
    # parse command line arguments
    all_db_names = ("mitdb", "vfdb", "cudb")
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--db-names", type=str, nargs="+", choices=all_db_names, default=all_db_names)
    parser.add_argument("-l", "--label-output", type=str, required=True)
    parser.add_argument("-f", "--feature-output", type=str)
    parser.add_argument("-s", "--segment-duration", type=int, default=8)
    parser.add_argument("-r", "--sampling-rate", type=int, default=250)
    parser.add_argument("-j", "--jobs", type=int, default=-1)
    # parser.add_argument("-m", "--label-method", type=int, default=-1)
    args = parser.parse_args()

    x_data_info = []
    x_data = []
    y_data = array('i')
    if not args.feature_output:  # only want to perform segmentation and labeling
        for idx, segment in load_all_segments(args.db_names, args.segment_duration):
            x_data_info.append(segment.info)
            y_data.append(label_segment(segment.info))
    else:  # perform segmentation + feature extraction
        parellel = Parallel(n_jobs=args.jobs, verbose=0, backend="multiprocessing", max_nbytes=2048)
        results = parellel(delayed(extract_features)(idx, segment, args.sampling_rate) for idx, segment in load_all_segments(args.db_names, args.segment_duration))
        # sort the results from multiple jobs according to the order they are emitted
        results.sort(key=lambda result: result[0])
        for idx, features, label, segment_info in results:
            x_data_info.append(segment_info)
            x_data.append(features)
            y_data.append(label)

    # write to output files
    # label/info file
    with open(args.label_output, "wb") as f:
        pickle.dump(x_data_info, f)
        pickle.dump(y_data, f)
        
    # features file
    if args.feature_output:
        with open(args.feature_output, "wb") as f:
            pickle.dump(x_data, f)

    # output summary
    n_all_artifact = 0
    for info in x_data_info:
        if info.has_artifact:
            n_all_artifact += 1
    print("\nall segments:", len(y_data), ", all vf:", np.sum(y_data), ", all artifacts:", n_all_artifact)


if __name__ == "__main__":
    main()
