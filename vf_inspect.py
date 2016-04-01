#!/usr/bin/env python2
import vf_features
import cPickle as pickle  # python 2 only
import numpy as np
import matplotlib.pyplot as plt
import sys


def main():
    # load errors
    error_log = "gb_errors.txt"
    errors = []
    with open(error_log, "r") as f:
        for line in f:
            errors.append(int(line.strip()))

    segments_cache_name = "all_segments.dat"
    # load cached segments if they exist
    try:
        i_err = 0
        error_idx = errors[0]
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
                            error_idx = errors[i_err]
                        i_seg += 1
                else:
                    break
                if show:
                    print "error:", segment.record, ", sample rate:", segment.sampling_rate, ", sample #:", segment.begin_time, ", has Vf:", segment.has_vf
                    # plt.plot(segment.signals)
                    # plt.show()
                    features = vf_features.extract_features(segment.signals, segment.sampling_rate, plotting=True)
                    print features
    except Exception:
        print sys.exc_info()
        pass
if __name__ == '__main__':
    main()
