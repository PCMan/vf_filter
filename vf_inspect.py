#!/usr/bin/env python3
import pyximport; pyximport.install()
import vf_features
import vf_data
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
import csv


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--db-names", type=str, nargs="+", default=None)
    # parser.add_argument("-o", "--output", type=str)
    parser.add_argument("-s", "--segment-duration", type=int, default=8)
    args = parser.parse_args()



    segments_cache_name = "all_segments.dat"
    # load cached segments if they exist
    try:
        i_err = 0
        error_idx = common_errors[0]
        i_seg = 0
        with open(segments_cache_name, "rb") as cache_file:
            while True:
                show = False
                segment = pickle.load(cache_file)
                if segment:
                    if not segment.has_artifact:  # ignore segments with artifacts
                        if i_seg == error_idx:
                            if np.random.choice((True, False), p=(0.2, 0.8)):
                                show = True
                            # get next error
                            i_err += 1
                            error_idx = common_errors[i_err]
                        i_seg += 1
                else:
                    break
                if show:
                    print("error:", segment.record, ", sample rate:", segment.sampling_rate, ", sample #:", segment.begin_time, ", has Vf:", segment.has_vf)
                    features = vf_features.extract_features(segment.signals, segment.sampling_rate)
                    for name, feature in zip(vf_features.feature_names, features):
                        print("{0}: {1}".format(name, feature))
                    print("-----------------------------------------")
                    # plot the signals
                    vf_features.preprocessing(segment.signals, segment.sampling_rate, plotting=True)
    except Exception:
        print(sys.exc_info())
        pass


if __name__ == '__main__':
    main()
