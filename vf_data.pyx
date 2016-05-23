#!/usr/bin/env python3
import pyximport; pyximport.install()
import numpy as np
import os
from array import array
import wfdb_reader


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


rhythm_name_tab = """
(AFIB	atrial fibrillation
(AF	atrial fibrillation
(ASYS	asystole
(B	ventricular bigeminy
(BI	first degree heart block
(HGEA	high grade ventricular ectopic activity
(N	normal sinus rhythm
(NSR	normal sinus rhythm
(NOD	nodal ("AV junctional") rhythm
(NOISE	noise
(PM	pacemaker (paced rhythm)
(SBR	sinus bradycardia
(SVTA	supraventricular tachyarrhythmia
(VER	ventricular escape rhythm
(VF	ventricular fibrillation
(VFL	ventricular flutter
(VT ventricular tachycardia
(AB	Atrial bigeminy
(AFIB		Atrial fibrillation
(AFL		Atrial flutter
(B		Ventricular bigeminy
(BII		2° heart block
(IVR		Idioventricular rhythm
(N		Normal sinus rhythm
(NOD		Nodal (A-V junctional) rhythm
(P		Paced rhythm
(PREX		Pre-excitation (WPW)
(SBR		Sinus bradycardia
(SVTA		Supraventricular tachyarrhythmia
(T		Ventricular trigeminy
(B3	Third degree heart block
(SAB	Sino-atrial block
"""
rhythm_descriptions = {}
for line in rhythm_name_tab.strip().split("\n"):
    cols = line.split("\t", maxsplit=1)
    if len(cols) > 1:
        name = cols[0].strip()
        desc = cols[1].strip()
        rhythm_descriptions[name] = desc


class Rhythm:
    def __init__(self, str name, int begin_time):
        self.name = name
        self.begin_time = begin_time
        self.end_time = 0
        self.beats = array("i")  # valid only when beat annotations are available

    def get_desc(self):
        return rhythm_descriptions.get(self.name, self.name)


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
        self.adc_zero = 0

    def load(self, str db_name, str record, int channel, str annotator):
        record_name = "{0}/{1}".format(db_name, record)
        self.name = record_name

        cdef list annotations
        (n_channels, self.sampling_rate, self.gain, self.adc_zero, signals) = wfdb_reader.read_signals(record_name, channel=channel)
        # print(record_name)

        # substract ADC zero and convert the unit to mV
        self.signals = (signals - self.adc_zero).astype("float64") / self.gain

        # read annotations
        annotations = []
        for ann in wfdb_reader.read_annotations(record_name, ann_name=annotator):
            annotation = Annotation(*ann)
            annotations.append(annotation)
        self.annotations = annotations

    def get_total_time(self):
        return len(self.signals) / self.sampling_rate

    # parse the annotations of the record_name to get all of the rhythms with artifacts excluded
    def get_artifact_free_rhythms(self):
        rhythms = []
        cdef list annotations = self.annotations
        cdef int n_beats = 0  # beat number of current rhythm
        cdef bint is_mghdb = self.name.startswith("mghdb/")
        cdef bint in_artifacts = False
        cdef str rhythm_type = ""
        # FIXME: create NSR rhythm at the beginning?
        current_rhythm = None
        for ann in annotations:
            if is_mghdb:  # mghdb uses its own annotation format and requires special handling
                if ann.is_beat_annotation():  # if this is a beat annotation
                    if current_rhythm:
                        current_rhythm.beats.append(ann.time)
                else:  # non-beat annotations (rhythm, quality, ...etc.)
                    if ann.code == '"':  # comment annotation (used for rhythm change)
                        rhythm_type = ann.get_rhythm_type()
                        if rhythm_type:
                            if current_rhythm:  # end of current rhythm
                                current_rhythm.end_time = ann.time
                                current_rhythm = None
                            # FIXME: how to handle artifacts in mghdb?
                            current_rhythm = Rhythm(rhythm_type, ann.time)
                            rhythms.append(current_rhythm)
            else:  # mitdb like formats (mitdb, vfdb, cudb, edb...)
                if ann.is_beat_annotation():  # if this is a beat annotation
                    if current_rhythm:
                        current_rhythm.beats.append(ann.time)
                else:  # non-beat annotations (rhythm, quality, ...etc.)
                    code = ann.code
                    if code == "+":  # rhythm change detected
                        if current_rhythm:  # end of current rhythm
                            current_rhythm.end_time = ann.time
                            current_rhythm = None

                        rhythm_type = ann.get_rhythm_type()
                        if not rhythm_type.startswith("(NOISE"):
                            current_rhythm = Rhythm(rhythm_type, ann.time)
                            rhythms.append(current_rhythm)
                    elif code == "[":  # begin of flutter/fibrillation found
                        if current_rhythm:  # end of current rhythm
                            current_rhythm.end_time = ann.time
                            current_rhythm = None

                        # FIXME: some records in cudb only has [ and ] and do not distinguish VF and VFL
                        # Let's label all of them as VF at the moment
                        rhythm_type = "(VF"
                        current_rhythm = Rhythm(rhythm_type, ann.time)
                        rhythms.append(current_rhythm)
                    elif code == "]":  # end of flutter/fibrillation found
                        if current_rhythm:  # end of current rhythm
                            current_rhythm.end_time = ann.time
                            current_rhythm = None
                    elif code == "!":  # ventricular flutter wave (this annotation is used by mitdb for V flutter beats)
                        pass
                    elif code == "|":  # isolated artifact
                        if current_rhythm:  # end of current rhythm (skip artifact)
                            current_rhythm.end_time = ann.time
                            current_rhythm = None
                    elif code == "~":  # change in quality
                        if ann.sub_type != 0:  # has noise or unreadable
                            if current_rhythm:  # end current rhythm to skip noise
                                current_rhythm.end_time = ann.time
                                current_rhythm = None
                        else:  # normal quality restored
                            if rhythm_type:  # continue from last rhythm type
                                current_rhythm = Rhythm(rhythm_type, ann.time)
                                rhythms.append(current_rhythm)
        # FIXME: merge rhythm segments of the same type if their are no gaps among them.
        if current_rhythm:  # end the last rhythm
            current_rhythm.end_time = len(self.signals)
        return rhythms


# info of a sample segment
class SegmentInfo:
    def __init__(self, record, rhythm, int begin_time, int end_time) -> object:
        self.record_name = record.name
        self.sampling_rate = record.sampling_rate
        self.begin_time = begin_time  # in terms of sample number (reletive to the head of the segment)
        self.end_time = end_time
        self.n_beats = 0  # number of beats, if available in the annotation
        if rhythm.beats:
            begin_beat_idx = np.searchsorted(rhythm.beats, begin_time, side="right")
            end_beat_idx = np.searchsorted(rhythm.beats, end_time, side="left")
            self.n_beats = (end_beat_idx - begin_beat_idx) + 1
        self.rhythm = rhythm.name

    def get_duration(self):
        return float(self.end_time - self.begin_time) / self.sampling_rate

    # valid only when beat annotations are available
    def get_heart_rate(self):
        cdef double duration = self.get_duration()
        cdef double n_beats = self.n_beats
        cdef double heart_rate = (n_beats / duration) * 60 if duration else 0.0  # heart rate (beats per minute)
        return heart_rate


# sample segment
class Segment:
    def __init__(self, info, signals) -> object:
        self.info = info
        self.signals = signals


# dataset used for AED testing
# composed rhythms from the following data sources:
# mitdb: all rhythms
# vfdb: all rhythms except VT
# cudb: all rhythms except NSR
# edb: all rhythms, especially VT
# mghdb: only VF and VT
class DataSet:
    def __init__(self):
        self.corrections = {}

    # Some rhythm annotations in the original datasets are incorrect, so we provide a mechanism to override them.
    def load_correction(self, str correction_file):
        corrections = {}
        try:
            with open(correction_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 3:
                        continue
                    record_name, begin_time, correct_rhythm = parts
                    corrections[(record_name, int(begin_time))] = correct_rhythm
        except IOError:
            print("Fail to open", correction_file)
        self.corrections = corrections

    # generator for ECG sample segments
    def get_samples(self, double duration=5.0):
        # nearly all rhythms other than VF are annotated as NSR in cudb
        # so it's unreliable. Drop NSR beats from cudb.
        sources = [
            ("mitdb", lambda rhythm: True),
            ("vfdb", lambda rhythm: rhythm.name != "(VT"),  # VT in vfdb does not contain beat annotations so rate is unknown
            ("cudb", lambda rhythm: rhythm.name != "(N"),  # NSR annotations in cudb are unreliable
            ("edb", lambda rhythm: True),
            ("mghdb", self.check_mghdb_rhythm)  # we only want VF and VT in mghdb
        ]

        cdef int segment_size, segment_begin, segment_end
        for db_name, check_rhythm in sources:
            if db_name == "mghdb":  # we only use these records from mghdb which contain VF and VT rhythms
                record_names = ["mgh040", "mgh041", "mgh229", "mgh236", "mgh044", "mgh046", "mgh122"]
                channel = 1  # lead II is at channel 1 in mghdb
                annotator = "ari"  # mghdb only contains ari annotations
            else:  # mitdb, vfdb, cudb, edb...etc.
                record_names = get_records(db_name)
                channel = 0
                annotator = "atr"

            corrections = self.corrections
            for record_name in record_names:
                record = Record()
                record.load(db_name, record_name, channel=channel, annotator=annotator)
                for rhythm in record.get_artifact_free_rhythms():
                    if check_rhythm(rhythm):  # if we should enroll this rhythm type
                        # print(db_name, record_name, rhythm.name)
                        # perform segmentation for this rhythm
                        segment_size = int(np.round(duration * record.sampling_rate))
                        for segment_begin in range(rhythm.begin_time, rhythm.end_time - segment_size, segment_size):
                            segment_end = segment_begin + segment_size
                            signals = record.signals[segment_begin:segment_end]
                            segment_info = SegmentInfo(record, rhythm, segment_begin, segment_end)

                            # check if we have corrections for this segment
                            correction = corrections.get((segment_info.record_name, segment_begin), None)
                            # "C" means correct and confirmed so there is no need to fix it if the mark is "C".
                            if correction and correction != "C":  # found an entry for the sample
                                # print("Fix", segment_info.record_name, segment_begin, correction)
                                # fix the incorrect rhythm annotation for this sample
                                segment_info.rhythm = correction
                            segment = Segment(segment_info, signals)
                            yield segment

    @staticmethod
    def check_mghdb_rhythm(rhythm):
        name = rhythm.name.strip().lower()
        if name == "vt" or name.startswith("ventricular tac"):
            rhythm.name = "(VT"
            return True
        elif name.startswith("ventricular fib"):
            rhythm.name = "(VF"
            return True
        return False
