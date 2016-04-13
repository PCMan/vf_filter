#!/usr/bin/env python3
import pyximport; pyximport.install()
from ctypes import *
import numpy as np
import scipy.signal
import os
from vf_features import extract_features
import pickle
from joblib import Parallel, delayed
from joblib.pool import has_shareable_memory
from array import array


DEFAULT_SAMPLING_RATE = 250.0
# dataset_dir = os.path.join(os.path.dirname(__file__), "datasets")
dataset_dir = "datasets"


# get name of records from a database
def get_records(db_name):
    records = []
    dirname = os.path.join(dataset_dir, db_name)
    for name in os.listdir(dirname):
        records.append(name)
    records.sort()
    return records


class Segment:
    def __init__(self, record, sampling_rate, signals, begin_time):
        self.record = record
        self.sampling_rate = sampling_rate
        self.signals = signals
        self.begin_time = begin_time  # in terms of sample number
        self.has_vf = False
        self.has_artifact = False


class Annotation:
    def __init__(self, time, code, sub_type, rhythm_type):
        self.time = time
        self.code = code
        self.sub_type = sub_type
        self.rhythm_type = rhythm_type


class Record:
    def __init__(self):
        self.signals = array("I")  # unsigned int
        self.annotations = []
        self.name = ""
        self.sampling_rate = 0

    def load(self, db_name, record):
        record_name = "{0}/{1}".format(db_name, record)
        self.name = record_name

        record_filename = os.path.join(dataset_dir, db_name, record)
        with open(record_filename, "rb") as f:
            self.sampling_rate = pickle.load(f)
            self.signals = pickle.load(f)

            # read annotations
            annotations = []
            for ann in pickle.load(f):
                annotation = Annotation(*ann)
                annotations.append(annotation)
            self.annotations = annotations

    def get_total_time(self):
        return len(self.signals) / self.sampling_rate

    # perform segmentation
    def get_segments(self, duration=8.0):
        n_samples = len(self.signals)
        segment_size = int(self.sampling_rate * duration)
        n_segments = int(np.floor(n_samples / segment_size))
        segments = []

        annotations = self.annotations
        n_annotations = len(annotations)
        i_ann = 0
        in_vf_episode = False
        in_artifacts = False
        for i_seg in range(n_segments):
            # split the segment
            segment_begin = i_seg * segment_size
            segment_end = segment_begin + segment_size

            segment_signals = self.signals[segment_begin:segment_end]
            segment = Segment(record=self.name, sampling_rate=self.sampling_rate, signals=segment_signals, begin_time=segment_begin)

            segment.has_vf = in_vf_episode  # label of the segment
            segment.has_artifact = in_artifacts
            vf_begin_time = 0
            vf_duration = 0
            # handle annotations belonging to this segment
            while i_ann < n_annotations:
                ann = annotations[i_ann]
                if ann.time < segment_end:
                    code = ann.code
                    rhythm_type = ann.rhythm_type
                    if in_vf_episode:  # current rhythm is Vf
                        if code == "]":  # end of Vf found
                            in_vf_episode = False
                            vf_end_time = ann.time
                            vf_duration += (vf_end_time - vf_begin_time)
                        elif code == "+" and not rhythm_type.startswith("(V"):  # end of Vf found
                            in_vf_episode = False
                            vf_end_time = ann.time
                            vf_duration += (vf_end_time - vf_begin_time)
                    else:  # current rhythm is not Vf
                        if code == "[":  # begin of Vf found
                            segment.has_vf = in_vf_episode = True
                            vf_begin_time = ann.time
                        elif code == "+" and rhythm_type.startswith("(V"):  # begin of Vf found
                            segment.has_vf = in_vf_episode = True
                            vf_begin_time = ann.time

                    if code == "+":  # change of rhythm
                        if rhythm_type.startswith("(NOISE"):  # the following signals are noise
                            segment.has_artifact = in_artifacts = True
                        else:  # change to some kind of rhythm other than noise
                            in_artifacts = False

                    if code == "!":  # ventricular flutter wave (this annotation is used by mitdb for V flutter beats)
                        segment.has_vf = True
                        vf_duration += self.sampling_rate  # add 1 second to vf_duration calculation
                    elif code == "|":  # isolated artifact
                        segment.has_artifact = True
                    elif code == "~":  # change in quality
                        if ann.sub_type != 0:  # has noise or unreadable
                            segment.has_artifact = in_artifacts = True
                        else:
                            in_artifacts = False

                    # if total Vf duration in this segment is less than 1 second, label it as non-Vf
                    # if segment.has_vf and vf_duration < self.sampling_rate * 1:
                    #     segment.has_vf = False

                    i_ann += 1
                else:
                    break

            # label of the segment as Vf only if Vf still persists at the end of the segment
            segment.has_vf = in_vf_episode

            segments.append(segment)
        return segments


# implement as a generator for ECG segments
def load_all_segments():
    segments_cache_name = "all_segments.dat"
    segment_duration = 8  # 8 sec per segment
    loaded_from_cache = False

    # load cached segments if they exist
    try:
        with open(segments_cache_name, "rb") as cache_file:
            loaded_from_cache = True
            while True:
                segment = pickle.load(cache_file)
                if segment:
                    # only do feature extration for segments without artifacts
                    if not segment.has_artifact:
                        yield segment
                else:
                    break
    except Exception:
        pass

    if not loaded_from_cache:
        try:
            cache_file = open(segments_cache_name, "wb")
        except Exception:
            cache_file = None

        # mitdb and vfdb contain two channels, but we only use the first one here
        # data source sampling rate:
        # mitdb: 360 Hz
        # vfdb, cudb: 250 Hz
        output = open("summary.csv", "w")
        output.write('"db", "record", "vf", "non-vf"\n')
        for db_name in ("mitdb", "vfdb", "cudb"):
            for record_name in get_records(db_name):
                print(("read record:", db_name, record_name))
                record = Record()
                record.load(db_name, record_name)
                # print("  sample rate:", record.sampling_rate, "# of samples:", len(record.signals), ", # of anns:", len(record.annotations))

                segments = record.get_segments(segment_duration)

                n_vf = np.sum([1 if segment.has_vf else 0 for segment in segments])
                n_non_vf = len(segments) - n_vf
                output.write('"{0}","{1}",{2},{3}\n'.format(db_name, record_name, n_vf, n_non_vf))
                print("  segments:", (n_vf + n_non_vf), "# of vf segments (label=1):", n_vf)

                for segment in segments:
                    if cache_file:  # cache the segment
                        pickle.dump(segment, cache_file)
                    if segment.has_artifact:  # exclude segments with artifacts
                        continue
                    yield segment
        output.close()
        if cache_file:
            pickle.dump(None, cache_file)
            cache_file.close()


def extract_features_job(segment):
    segment_duration = 8  # 8 sec per segment
    # resample to DEFAULT_SAMPLING_RATE as needed
    signals = segment.signals
    if segment.sampling_rate != DEFAULT_SAMPLING_RATE:
        signals = scipy.signal.resample(signals, DEFAULT_SAMPLING_RATE * segment_duration)
    features = extract_features(signals, DEFAULT_SAMPLING_RATE)
    label = 1 if segment.has_vf else 0
    print(segment.record, segment.begin_time, features, label)
    return (features, label, segment.record, segment.begin_time)


def load_data(n_jobs):
    features_cache_name = "features.dat"
    x_info = []
    # load cached features if they exist
    try:
        with open(features_cache_name, "rb") as f:
            x_data = pickle.load(f)
            y_data = pickle.load(f)
            x_info = pickle.load(f)
    except Exception:
        # load segments and perform feature extraction
        # here we use multiprocessing for speed up.
        features = Parallel(n_jobs=n_jobs, verbose=1, backend="multiprocessing", max_nbytes=4096)(delayed(extract_features_job)(seg) for seg in load_all_segments())

        # receive extracted features from the worker processes
        x_data = []
        y_data = []
        for item in features:
            (feature, label, record, begin_time) = item
            x_data.append(feature)
            y_data.append(label)
            # store mapping of feature and the segment it's built from
            x_info.append((record, begin_time))
        x_data = np.array(x_data)
        y_data = np.array(y_data)

        # cache the data
        try:
            with open(features_cache_name, "wb") as f:
                pickle.dump(x_data, f)
                pickle.dump(y_data, f)
                pickle.dump(x_info, f)
        except Exception:
            pass
        print("features are extracted.")

    return x_data, y_data, np.array(x_info)
