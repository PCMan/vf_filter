#!/usr/bin/env python3
import pyximport; pyximport.install()
import numpy as np
import os
from array import array
import wfdb_read

# dataset_dir = os.path.join(os.path.dirname(__file__), "datasets")
filelist_dir = "file_lists"


# get name of records from a database
def get_records(str db_name):
    records = []
    with open(os.path.join(filelist_dir, "{0}.txt".format(db_name)), "r") as f:
        for line in f:
            name = line.strip()
            records.append(name)
    records.sort()
    return records


class Rhythm:
    def __init__(self, str name, int begin_time, int begin_beat):
        self.name = name
        self.begin_time = begin_time
        self.begin_beat = begin_beat  # valid when beat annotations are available
        self.n_beats = 0  # valid when beat annotations are available
        self.duration = 0.0

    # valid only when beat annotations are available
    def get_heart_rate(self):
        cdef double heart_rate = (self.n_beats / self.duration) * 60 if self.duration else 0.0  # heart rate (beats per minute)
        return heart_rate


class SegmentInfo:
    def __init__(self, str record, int sampling_rate, int begin_time) -> object:
        self.record = record
        self.sampling_rate = sampling_rate
        self.begin_time = begin_time  # in terms of sample number (reletive to the head of the segment)
        self.has_artifact = False
        self.n_beats = 0  # number of beats, if available in the annotation
        self.rhythms = []

    def has_rhythm(self, str name):
        for rhythm in self.rhythms:
            if rhythm.name == name:
                return True
        return False

    def add_rhythm(self, str name, int begin_time, int begin_beat):
        self.rhythms.append(Rhythm(name, begin_time, begin_beat))

    def has_transition(self):
        return len(self.rhythms) > 1

    def get_last_rhythm_name(self):
        return self.rhythms[-1].name if self.rhythms else ""

    def get_last_rhythm(self):
        return self.rhythms[-1] if self.rhythms else None


class Segment:
    def __init__(self, record, info, signals) -> object:
        self.info = info
        self.signals = signals

        gain = record.gain # use to convert signal values to mV
        cdef int i = 0
        cdef int n_rhythms = len(info.rhythms)
        cdef int end_time, end_beat
        cdef double amplitude
        # calculate some info for each rhythm
        for i in range(n_rhythms):
            rhythm = info.rhythms[i]
            if i < n_rhythms:
                next_rhythm = info.rhythms[i + 1] if (i + 1) < n_rhythms else None

            end_time = next_rhythm.begin_time if next_rhythm else len(signals)
            rhythm.duration = float(end_time - rhythm.begin_time) / info.sampling_rate  # duration in terms of second
            if info.n_beats > 0:
                end_beat = next_rhythm.begin_beat if next_rhythm else info.n_beats
                rhythm.n_beats = end_beat - rhythm.begin_beat

            if rhythm.name == "(VF":  # try to distinguish coarse VF from fine VF
                # References for the definition of "coarse":
                # 1. Foundations of Respiratory Care. by Kenneth A. Wyka，Paul J. Mathews，John Rutkowski
                #    Chapter 19. p.537
                #    Quote: "Coarse VF exists when wave amplitude is more than 3 mm."
                # 2. ECGs Made Easy by Barbara J Aehlert
                #    p.203
                #    Quote: "Coarse VF is 3 mm or more in amplitude. Fine VF is less than 3 mm in amplitude."
                # 3. In AHA recommendations for AED, a peak-to-peak amplitude of 0.2 mV is suggested.
                # print(rhythm.name, rhythm.begin_time, end_time)
                if end_time > rhythm.begin_time:
                    rhythm_signals = signals[rhythm.begin_time:end_time]
                    # FIXME: max - min only gives a rough estimate of peak-to-peak amplitude here :-(
                    amplitude = (np.max(rhythm_signals) - np.min(rhythm_signals)) / gain

                    # in normal ECG settings, amplitude of 1 mm means 0.1 mV
                    # Here we use the threshold 0.2 mV suggested by AHA despite that some other books use 0.3 mV instead.
                    rhythm.is_coarse = True if amplitude > 0.2 else False
                else:
                    rhythm.is_coarse = False
                # if not rhythm.is_coarse:
                #     print("Fine VF")


beat_type_desc = """
N		Normal beat (displayed as "·" by the PhysioBank ATM, LightWAVE, pschart, and psfd)
L		Left bundle branch block beat
R		Right bundle branch block beat
B		Bundle branch block beat (unspecified)
A		Atrial premature beat
a		Aberrated atrial premature beat
J		Nodal (junctional) premature beat
S		Supraventricular premature or ectopic beat (atrial or nodal)
V		Premature ventricular contraction
r		R-on-T premature ventricular contraction
F		Fusion of ventricular and normal beat
e		Atrial escape beat
j		Nodal (junctional) escape beat
n		Supraventricular escape beat (atrial or nodal)
E		Ventricular escape beat
/		Paced beat
f		Fusion of paced and normal beat
Q		Unclassifiable beat
?		Beat not classified during learning
"""
beat_types = {}
for line in beat_type_desc.strip().split("\n"):
    cols = line.split("\t", maxsplit=1)
    beat_types[cols[0].strip()] = cols[1].strip()


rhythm_alias = {
    "(VFIB": "(VF",
    "(AFIB": "(AF",
    "(NSR": "(N",
    "(PM": "(P"
}


class Annotation:
    def __init__(self, int time, str code, int sub_type, str rhythm_type) -> object:
        self.time = time
        self.code = code
        self.sub_type = sub_type
        self.rhythm_type = rhythm_type

    def get_rhythm_type(self):
        return rhythm_alias.get(self.rhythm_type, self.rhythm_type)

    def is_beat_annotation(self):
        return self.code in beat_types


class Record:
    def __init__(self) -> object:
        self.signals = array("I")  # unsigned int
        self.annotations = []
        self.name = ""
        self.sampling_rate = 0
        self.gain = 0

    def load(self, str db_name, str record):
        record_name = "{0}/{1}".format(db_name, record)
        self.name = record_name

        cdef list annotations
        (n_channels, self.sampling_rate, self.gain, self.signals) = wfdb_read.read_signals(record_name)
        # print(record_name)

        # read annotations
        annotations = []
        for ann in wfdb_read.read_annotations(record_name):
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
        cdef bint in_artifact = False
        cdef int segment_begin, segment_end
        cdef str rhythm_type = ""
        for segment_begin in range(0, n_samples, segment_size):
            # split the segment
            segment_end = segment_begin + segment_size
            segment_signals = self.signals[segment_begin:segment_end]
            segment_info = SegmentInfo(record=self.name, sampling_rate=self.sampling_rate, begin_time=segment_begin)
            segment_info.has_artifact = in_artifacts

            n_beats = 0  # beat number of current rhythm
            if rhythm_type:
                segment_info.add_rhythm(rhythm_type, begin_time=0, begin_beat=0)
            # handle annotations belonging to this segment
            while i_ann < n_annotations:
                ann = annotations[i_ann]
                if ann.time < segment_end:
                    if ann.is_beat_annotation():  # if this is a beat annotation
                        n_beats += 1
                    else:  # non-beat annotations
                        code = ann.code
                        if code == "+":  # rhythm change detected
                            rhythm_type = ann.get_rhythm_type()
                            if rhythm_type:
                                segment_info.add_rhythm(rhythm_type, begin_time=(ann.time - segment_begin), begin_beat=n_beats)

                            if rhythm_type.startswith("(NOISE"):
                                segment_info.has_artifact = in_artifacts = True
                            # print(self.name, "NOISE found", ann.time)
                            else:
                                in_artifacts = False
                        elif code == "[":  # begin of flutter/fibrillation found
                            # FIXME: some records in cudb only has [ and ] and do not distinguish VF and VFL
                            # Let's label all of them as VF at the moment
                            # print(self.name, "[ found", ann.time)
                            rhythm_type = "(VF"
                            segment_info.add_rhythm(rhythm_type, begin_time=(ann.time - segment_begin), begin_beat=n_beats)
                        elif code == "]":  # end of flutter/fibrillation found
                            # print(self.name, "] found", ann.time)
                            # print("end of VF/VFL", state)
                            rhythm_type = ""
                        elif code == "!":  # ventricular flutter wave (this annotation is used by mitdb for V flutter beats)
                            segment_info.add_rhythm("(VFL", begin_time=(ann.time - segment_begin), begin_beat=n_beats)
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

            segment_info.n_beats = n_beats
            segment = Segment(self, info=segment_info, signals=segment_signals)
            yield segment  # generate a new segment instance

