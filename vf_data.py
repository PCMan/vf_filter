#!/usr/bin/env python2
from ctypes import *
import numpy as np
import scipy.signal
import os
import wfdb
from vf_features import extract_features
# import pickle
import cPickle as pickle  # python 2 only
from multiprocessing import Pool


DEFAULT_SAMPLING_RATE = 360.0
N_JOBS = 4


# get name of records from a database
def get_records(db_name):
    records = []
    dirname = os.path.expanduser("~/database/{0}".format(db_name))
    for name in os.listdir(dirname):
        if name.endswith(".dat"):
            record = os.path.splitext(name)[0]
            records.append(record)
    records.sort()
    return records


class Segment:
    def __init__(self, record, sampling_rate, signals):
        self.record = record
        self.sampling_rate = sampling_rate
        self.signals = signals
        self.has_vf = False
        self.has_artifact = False


class Annotation:
    # ann is an instance of wfdb.WFDB_Annotation
    def __init__(self, ann):
        self.time = ann.time
        ann_code = ord(ann.anntyp)
        self.code = wfdb.annstr(ann_code)
        self.sub_type = ord(ann.subtyp)
        self.rhythm_type = ""
        if ann.aux:
            # the first byte of aux is the length of the string
            aux_ptr = cast(ann.aux, c_void_p).value + 1  # skip the first byte
            self.rhythm_type = cast(aux_ptr, c_char_p).value


class Record:
    def __init__(self):
        self.signals = []
        self.annotations = []
        self.name = ""
        self.sampling_rate = 0

    def load(self, db_name, record):
        record_name = "{0}/{1}".format(db_name, record)
        self.name = record_name

        # query number of channels in this record
        n_channels = wfdb.isigopen(record_name, None, 0)

        # query sampling rate of the record
        self.sampling_rate = wfdb.sampfreq(record_name)

        # read the signals
        sigInfo = (wfdb.WFDB_Siginfo * n_channels)()
        sample_buf = (wfdb.WFDB_Sample * n_channels)()
        signals = []
        if wfdb.isigopen(record_name, byref(sigInfo), n_channels) == n_channels:
            while wfdb.getvec(byref(sample_buf)) > 0:
                sample = sample_buf[0]  # we only want the first channel
                signals.append(sample)
        signals = np.array(signals)

        # read annotations
        annotations = []
        ann_name = wfdb.String("atr")
        ann_info = (wfdb.WFDB_Anninfo * n_channels)()
        for item in ann_info:
            item.name = ann_name
            item.stat = wfdb.WFDB_READ
        if wfdb.annopen(record_name, byref(ann_info), n_channels) == 0:
            ann_buf = (wfdb.WFDB_Annotation * n_channels)()
            while wfdb.getann(0, byref(ann_buf)) == 0:
                ann = ann_buf[0]  # we only want the first channel
                annotation = Annotation(ann)
                annotations.append(annotation)

        self.signals = signals
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
            segment = Segment(record=self.name, sampling_rate=self.sampling_rate, signals=segment_signals)

            segment.has_vf = in_vf_episode  # label of the segment
            segment.has_artifact = in_artifacts
            # handle annotations belonging to this segment
            while i_ann < n_annotations:
                ann = annotations[i_ann]
                if ann.time < segment_end:
                    code = ann.code
                    rhythm_type = ann.rhythm_type
                    if in_vf_episode:  # current rhythm is Vf
                        if code == "]" or not rhythm_type.startswith("(V"):  # end of Vf found
                            in_vf_episode = False
                    else:  # current rhythm is not Vf
                        if code == "[" or rhythm_type.startswith("(V"):  # begin of Vf found
                            segment.has_vf = in_vf_episode = True
                    if code == "|":  # isolated artifact
                        segment.has_artifact = True
                    elif code == "~":  # change in quality
                        if ann.sub_type != 0:  # has noise or unreadable
                            segment.has_artifact = in_artifacts = True
                        else:
                            in_artifacts = False
                    i_ann += 1
                else:
                    break
            '''
            if segment.has_artifact:
                import matplotlib.pyplot as plt
                plt.plot(segment.signals)
                plt.show()
            '''
            segments.append(segment)
        return segments


def extract_features_job(s):
    db_name, record_name, segment, sample_rate = s
    return extract_features(segment, sampling_rate=DEFAULT_SAMPLING_RATE)


def load_all_segments():
    segments_cache_name = "all_segments.dat"
    segment_duration = 8  # 8 sec per segment
    all_segments = []
    all_labels = []
    # load cached segments if they exist
    try:
        with open(segments_cache_name, "rb") as f:
            all_segments = pickle.load(f)
            all_labels = pickle.load(f)
    except Exception:
        pass

    if not all_segments or not all_labels:
        # mitdb and vfdb contain two channels, but we only use the first one here
        # data source sampling rate:
        # mitdb: 360 Hz
        # vfdb, cudb: 250 Hz
        output = open("summary.csv", "w")
        output.write('"db", "record", "vf", "non-vf"\n')
        for db_name in ("mitdb", "vfdb", "cudb"):
            for record_name in get_records(db_name):
                print "read record:", db_name, record_name
                record = Record()
                record.load(db_name, record_name)

                print "  sample rate:", record.sampling_rate, "# of samples:", len(record.signals), ", # of anns:", len(record.annotations)

                segments = record.get_segments(segment_duration)

                n_vf = np.sum([1 if segment.has_vf else 0 for segment in segments])
                n_non_vf = len(segments) - n_vf
                output.write('"{0}","{1}",{2},{3}\n'.format(db_name, record_name, n_vf, n_non_vf))
                print "  segments:", (n_vf + n_non_vf), "# of vf segments (label=1):", n_vf

                for segment in segments:
                    if segment.has_artifact:  # exclude segments with artifacts
                        continue
                    # resample to DEFAULT_SAMPLING_RATE as needed
                    signals = segment.signals
                    if segment.sampling_rate != DEFAULT_SAMPLING_RATE:
                        signals = scipy.signal.resample(signals, DEFAULT_SAMPLING_RATE * segment_duration)
                    all_segments.append((db_name, record_name, signals, segment.sampling_rate))
                    all_labels.append(1 if segment.has_vf else 0)
        wfdb.wfdbquit()
        output.close()

        # cache the segments
        '''
        try:
            with open(segments_cache_name, "wb") as f:
                pickle.dump(all_segments, f)
                pickle.dump(all_labels, f)
        except Exception:
            pass
        '''
    print "segments:", len(all_segments), ", # of VT/Vf:", np.sum(all_labels)
    return all_segments, all_labels


def load_data():
    features_cache_name = "features.dat"
    x_data = []
    y_data = []
    # load cached segments if they exist
    try:
        with open(features_cache_name, "rb") as f:
            x_data = pickle.load(f)
            y_data = pickle.load(f)
    except Exception:
        all_segments, all_labels = load_all_segments()

        # use multiprocessing for speed up.
        print "start feature extraction..."
        pool = Pool(N_JOBS)
        x_data = pool.map(extract_features_job, all_segments)
        '''
        x_data = []
        for db_name, record_name, segment, sample_rate in all_segments:
            # convert segment values to features
            x_data.append(extract_features(segment, sampling_rate=sample_rate))
        '''
        x_data = np.array(x_data)
        y_data = np.array(all_labels)

        # cache the data
        try:
            with open(features_cache_name, "wb") as f:
                pickle.dump(x_data, f)
                pickle.dump(y_data, f)
        except Exception:
            pass
        print "features are extracted."

    return x_data, y_data
