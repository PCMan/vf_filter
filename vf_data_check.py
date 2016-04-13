#!/usr/bin/env python3
# check data integraty of the pickle files
import pyximport; pyximport.install()
import pickle


def main():
    segments_cache_name = "all_segments.dat"
    n_segments = 0
    n_artifacts = 0
    try:
        with open(segments_cache_name, "rb") as cache_file:
            while True:
                segment = pickle.load(cache_file)
                if not segment:
                    break
                n_segments += 1
                if segment.has_artifact:
                    n_artifacts += 1
    except Exception:
        pass
    print("segments:", n_segments, ", excluded:", n_artifacts)

    features_cache_name = "features.dat"
    # load cached features if they exist
    with open(features_cache_name, "rb") as f:
        x_data = pickle.load(f)
        y_data = pickle.load(f)
        x_info = pickle.load(f)
        print("features:", len(x_data), ", labels:", len(y_data))


if __name__ == '__main__':
    main()
