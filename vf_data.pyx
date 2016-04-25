#!/usr/bin/env python3
import pyximport; pyximport.install()
import numpy as np
import os
import pickle
from array import array


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
        self.terminating_rhythm = ""
        self.rhythm_types = set()

    def has_rhythm(self, name):
        return name in self.rhythm_types


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

    def get_rhythm_type(self):
        if self.rhythm_type == "(VFIB":
            return "(VF"
        elif self.rhythm_type == "(AFIB":
            return "(AF"
        return self.rhythm_type


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
        cdef int segment_begin, segment_end
        cdef str rhythm_type = ""
        for segment_begin in range(0, n_samples, segment_size):
            # split the segment
            segment_end = segment_begin + segment_size
            segment_signals = self.signals[segment_begin:segment_end]
            segment_info = SegmentInfo(record=self.name, sampling_rate=self.sampling_rate, begin_time=segment_begin)
            segment_info.has_artifact = in_artifacts
            if rhythm_type:
                segment_info.rhythm_types.add(rhythm_type)

            # handle annotations belonging to this segment
            while i_ann < n_annotations:
                ann = annotations[i_ann]
                if ann.time < segment_end:
                    code = ann.code
                    if code == "+":  # rhythm change detected
                        rhythm_type = ann.get_rhythm_type()
                        if rhythm_type:
                            segment_info.rhythm_types.add(rhythm_type)

                        if rhythm_type.startswith("(NOISE"):
                            segment_info.has_artifact = in_artifacts = True
                        # print(self.name, "NOISE found", ann.time)
                        else:
                            in_artifacts = False
                            has_transition = True
                    elif code == "[":  # begin of flutter/fibrillation found
                        # FIXME: some records in cudb only has [ and ] and do not distinguish VF and VFL
                        # Let's label all of them as VF at the moment
                        # print(self.name, "[ found", ann.time)
                        rhythm_type = "(VF"
                        segment_info.rhythm_types.add(rhythm_type)
                    elif code == "]":  # end of flutter/fibrillation found
                        # print(self.name, "] found", ann.time)
                        # print("end of VF/VFL", state)
                        rhythm_type = ""
                    elif code == "!":  # ventricular flutter wave (this annotation is used by mitdb for V flutter beats)
                        segment_info.rhythm_types.add("(VFL")
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
            segment_info.terminating_rhythm = rhythm_type
            segment = Segment(info=segment_info, signals=segment_signals)
            yield segment  # generate a new segment instance
