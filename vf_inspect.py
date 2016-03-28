#!/usr/bin/env python2
import vf_features
import cPickle as pickle  # python 2 only
import numpy as np
import matplotlib.pyplot as plt
import sys


def main():
    segments_cache_name = "all_segments.dat"
    # load cached segments if they exist
    try:
        with open(segments_cache_name, "rb") as cache_file:
            while True:
                show = False
                segment = pickle.load(cache_file)
                if segment:
                    if segment.has_vf:
                        if np.random.choice((True, False), p=(0.2, 0.8)):
                            show = True
                else:
                    break
                if show:
                    features = vf_features.extract_features(segment.signals, segment.sampling_rate, plotting=True)
                    print segment.record, segment.sampling_rate, features
                    plt.plot(segment.signals)
                    plt.show()
    except Exception:
        print sys.exc_info()
        pass
if __name__ == '__main__':
    main()
