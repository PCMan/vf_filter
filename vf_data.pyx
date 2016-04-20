#!/usr/bin/env python3
import pyximport; pyximport.install()
import numpy as np
import scipy.signal
import os
from vf_features import extract_features
import pickle
from joblib import Parallel, delayed
from joblib.pool import has_shareable_memory
from array import array


# constant values
DEFAULT_SAMPLING_RATE = 250

RHYTHM_NORMAL = 0
RHYTHM_VF = 1
RHYTHM_VT = 2
RHYTHM_VFL = 3
cdef int RHYTHM_ASYSTOLE = 4  # not used at the moment
cdef int RHYTHM_NOISE = 5
cdef int RHYTHM_UNKNOWN = 6


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


class SegmentInfo:
    def __init__(self, record, int sampling_rate, int begin_time) -> object:
        self.record = record
        self.sampling_rate = sampling_rate
        self.begin_time = begin_time  # in terms of sample number
        self.has_artifact = False
        self.has_transition = False
        self.has_vf = False
        self.has_vfl = False
        self.has_vt = False
        self.terminating_rhythm = RHYTHM_NORMAL


class Segment:
    def __init__(self, info, signals) -> object:
        self.info = info
        self.signals = signals


class Annotation:
    def __init__(self, int time, code, sub_type, rhythm_type) -> object:
        self.time = time
        self.code = code
        self.sub_type = sub_type
        self.rhythm_type = rhythm_type


class Record:
    def __init__(self) -> object:
        self.signals = array("I")  # unsigned int
        self.annotations = []
        self.name = ""
        self.sampling_rate = 0

    def load(self, db_name, record):
        record_name = "{0}/{1}".format(db_name, record)
        self.name = record_name

        record_filename = os.path.join(dataset_dir, db_name, record)
        cdef list annotations
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

    # perform segmentation (this is a python generator)
    def get_segments(self, double duration=8.0):
        cdef int n_samples = len(self.signals)
        cdef int segment_size = int(self.sampling_rate * duration)
        cdef list annotations = self.annotations
        cdef int n_annotations = len(annotations)
        cdef int i_ann = 0
        cdef bint has_transition = False, in_artifact = False
        cdef int current_rrhythm = RHYTHM_NORMAL
        cdef int segment_begin, segment_end
        for segment_begin in range(0, n_samples, segment_size):
            # split the segment
            segment_end = segment_begin + segment_size
            segment_signals = self.signals[segment_begin:segment_end]
            segment_info = SegmentInfo(record=self.name, sampling_rate=self.sampling_rate, begin_time=segment_begin)
            segment = Segment(info=segment_info, signals=segment_signals)
            segment_info.has_artifact = in_artifacts
            # handle annotations belonging to this segment
            while i_ann < n_annotations:
                ann = annotations[i_ann]
                if ann.time < segment_end:
                    code = ann.code
                    rhythm_type = ann.rhythm_type
                    if code == "+":  # rhythm change detected
                        if rhythm_type.startswith("(NOISE"):
                            in_artifacts = True
                            # print(self.name, "NOISE found", ann.time)
                        else:
                            in_artifacts = False
                            has_transition = True
                            if rhythm_type == "(VF" or rhythm_type == "(VFIB":
                                current_rrhythm = RHYTHM_VF
                                segment_info.has_vf = True
                                # print(self.name, "VF found", ann.time)
                            elif rhythm_type == "(VFL":
                                current_rrhythm = RHYTHM_VFL
                                segment_info.has_vfl = True
                                # print(self.name, "VFL found", ann.time)
                            elif rhythm_type == "(VT":
                                current_rrhythm = RHYTHM_VT
                                current_rrhythm = RHYTHM_NOISE
                                segment_info.has_vt = True
                                # print(self.name, "VT found", ann.time)
                            else:
                                current_rrhythm = RHYTHM_NORMAL
                                # print(self.name, "NSR found", ann.time)
                    elif code == "[":  # begin of flutter/fibrillation found
                        # FIXME: some records in cudb only has [ and ] and do not distinguish VF and VFL
                        # Let's label all of them as VF at the moment
                        # print(self.name, "[ found", ann.time)
                        current_rrhythm = RHYTHM_VF
                        segment_info.has_vf = True
                    elif code == "]":  # end of flutter/fibrillation found
                        # print(self.name, "] found", ann.time)
                        # print("end of VF/VFL", state)
                        current_rrhythm = RHYTHM_NORMAL  # FIXME: is this correct?
                    elif code == "!":  # ventricular flutter wave (this annotation is used by mitdb for V flutter beats)
                        segment_info.has_vfl = True
                    elif code == "|":  # isolated artifact
                        segment_info.has_artifact = True
                    elif code == "~":  # change in quality
                        if ann.sub_type != 0:  # has noise or unreadable
                            segment_info.has_artifact = in_artifacts = True
                        else:
                            in_artifacts = False
                    i_ann += 1
                else:
                    break
            # label of the segment as Vf only if Vf still persists at the end of the segment
            segment_info.terminating_rhythm = current_rrhythm
            yield segment  # generate a new segment instance


# implement as a generator for ECG segments
def load_all_segments():
    cdef double segment_duration = 8  # 8 sec per segment
    cdef bint loaded_from_cache = False

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

            n_vf = np.sum([1 if segment.info.terminating_rhythm == RHYTHM_VF else 0 for segment in segments])
            n_non_vf = len(segments) - n_vf
            output.write('"{0}","{1}",{2},{3}\n'.format(db_name, record_name, n_vf, n_non_vf))
            print("  segments:", (n_vf + n_non_vf), "# of vf segments (label=1):", n_vf)

            for segment in segments:
                if segment.info.has_artifact:  # exclude segments with artifacts
                    continue
                yield segment
    output.close()


def load_data(features_file, labels_file, exclude_artifact=True):
    x_data = []
    x_data_info = []
    y_data = []
    with open(features_file, "rb") as f:
        x_data = pickle.load(f)

    with open(labels_file, "rb") as f:
        x_data_info = pickle.load(f)
        y_data = pickle.load(f)

    x_data = np.array(x_data)
    y_data = np.array(y_data)
    x_data_info = np.array(x_data_info)

    if exclude_artifact:
        no_artifact_idx = [i for i, info in enumerate(x_data_info) if info.has_artifact == False]
        x_data = x_data[no_artifact_idx]
        y_data = y_data[no_artifact_idx]
        x_data_info = x_data_info[no_artifact_idx]

    return x_data, y_data, x_data_info
