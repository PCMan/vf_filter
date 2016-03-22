#!/usr/bin/env python2

from ctypes import *
import numpy as np
import scipy.signal
import os
import wfdb
import matplotlib.pyplot as plt 
from vf_features import extract_features
import pickle
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import ensemble
from sklearn import cross_validation
from sklearn import metrics
from sklearn import svm
from multiprocessing import Pool


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
    def __init__(self):
        self.record = ""
        self.label = 0
        self.signals = None
        self.begin_time = 0
        self.duration = 0


class Record:
    def __init__(self):
        self.signals = []
        self.annotations = []
        self.name = ""
        self.sample_rate = 0

    def load(self, db_name, record):
        record_name = "{0}/{1}".format(db_name, record)
        self.name = record_name

        # query number of channels in this record
        n_channels = wfdb.isigopen(record_name, None, 0)

        # query sampling rate of the record
        self.sample_rate = wfdb.sampfreq(record_name)

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
                ann_code = ord(ann.anntyp)
                rhythm_type = ""
                if ann.aux:
                    # the first byte of aux is the length of the string
                    aux_ptr = cast(ann.aux, c_void_p).value + 1  # skip the first byte
                    rhythm_type = cast(aux_ptr, c_char_p).value
                # print ann.time, wfdb.anndesc(ann_code), wfdb.annstr(ann_code), rhythm_type
                annotations.append((ann.time, wfdb.annstr(ann_code), rhythm_type))

        self.signals = signals
        self.annotations = annotations

    def get_vf_episodes(self):
        annotations = self.annotations
        n_samples = len(self.signals)
        episodes = []
        vf_begin = -1
        for (ann_time, code, rhythm_type) in annotations:
            if vf_begin > 0:  # current rhythm is Vf
                if code == "]" or not rhythm_type.startswith("(V"):
                    vf_end = ann_time
                    episodes.append((vf_begin, vf_end))
                    vf_begin = -1
            else:  # no Vf
                if code == "[" or rhythm_type.startswith("(V"):
                    vf_begin = ann_time
        if vf_begin > 0:
            episodes.append((vf_begin, n_samples))

        return episodes

    def get_total_time(self):
        return len(self.signals) / self.sample_rate

    # perform segmentation
    def get_segments(self, duration=8.0):
        n_samples = len(self.signals)
        segment_size = int(self.sample_rate * duration)
        n_segments = int(np.floor(n_samples / segment_size))
        segments = []
        labels = []

        vf_episodes = self.get_vf_episodes()
        n_episodes = len(vf_episodes)
        current_episode = 0
        for i in range(n_segments):
            # split the segment
            segment_begin = i * segment_size
            segment_end = segment_begin + segment_size
            segment = self.signals[segment_begin:segment_end]
            segments.append(segment)

            # try to label the segment
            has_vf = 0
            if current_episode < n_episodes:
                # check if the segment is overlapped with the current Vf episode
                (vf_begin, vf_end) = vf_episodes[current_episode]
                if vf_begin <= segment_begin <= vf_end:
                    has_vf = 1
                elif vf_begin <= segment_end <= vf_end:
                    has_vf = 1
                elif vf_begin >= segment_begin and vf_end <= segment_end:
                    has_vf = 1
                if segment_end >= vf_end:
                    current_episode += 1
            labels.append(has_vf)
        return np.array(segments), np.array(labels)

def extract_features_job(s):
    return extract_features(s[2], sampling_rate=360)

def main():
    cache_file_name = "all_segments.dat"
    segment_duration = 8  # 8 sec per segment
    all_segments = []
    all_labels = []
    # load cached segments if they exist
    try:
        with open(cache_file_name, "rb") as f:
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

                print "  sample rate:", record.sample_rate, "# of samples:", len(record.signals), ", # of anns:", len(record.annotations)

                segments, labels = record.get_segments(segment_duration)
                print "  segments:", len(segments), ", segment size:", len(segments[0])
                print "  # of vf segments (label=1):", np.sum(labels)

                n_vf = np.sum(labels)
                n_non_vf = len(segments) - n_vf
                output.write('"{0}","{1}",{2},{3}\n'.format(db_name, record_name, n_vf, n_non_vf))

                for segment in segments:
                    # resample to 360 Hz as needed (mainly for cudb)
                    if record.sample_rate != 360:
                        segment = scipy.signal.resample(segment, 360 * segment_duration)

                    all_segments.append((db_name, record_name, segment))
                all_labels.extend(labels)
                '''
                for segment, has_vf in zip(segments, labels):
                    if has_vf:
                        plt.plot(segment)
                        plt.show()
                '''
        wfdb.wfdbquit()
        output.close()

        # cache the segments
        try:
            with open(cache_file_name, "wb") as f:
                pickle.dump(all_segments, f)
                pickle.dump(all_labels, f)
        except Exception:
            pass

    print "Summary:\n", "# of segments:", len(all_segments), "# of VT/Vf:", np.sum(all_labels)

    # use multiprocessing for speed up.
    x_data = []
    pool = Pool(6)
    x_data = pool.map(extract_features_job, all_segments)
    '''
    for db_name, record_name, segment in all_segments:
        # convert segment values to features
        x_data.append(extract_features(segment, sampling_rate=360))
    '''
    x_data = np.array(x_data)
    y_data = np.array(all_labels)
    print "features are extracted."

    # normalize the features
    preprocessing.normalize(x_data)

    x_train, x_test, y_train, y_test = cross_validation.train_test_split(x_data, y_data, test_size=0.2, random_state=107)
    estimator = linear_model.LogisticRegression()
    estimator.fit(x_train, y_train)

    y_predict = estimator.predict(x_test)
    # print "Logistic regression: error:", float(np.sum(y_predict != y_test) * 100) / len(y_test), "%"
    print "Logistic regression: precision:\n", metrics.classification_report(y_test, y_predict), "\n"

    estimator = ensemble.RandomForestClassifier()
    estimator.fit(x_train, y_train)

    y_predict = estimator.predict(x_test)
    # print "RandomForest: error:", float(np.sum(y_predict != y_test) * 100) / len(y_test), "%"
    print "RandomForest:\n", metrics.classification_report(y_test, y_predict), "\n"

    estimator = svm.SVC(C=10, shrinking=False, cache_size=512, verbose=True)
    estimator.fit(x_train, y_train)

    y_predict = estimator.predict(x_test)
    print "SVC:\n", metrics.classification_report(y_test, y_predict), "\n"

    estimator = ensemble.GradientBoostingClassifier()
    estimator.fit(x_train, y_train)

    y_predict = estimator.predict(x_test)
    # print "RandomForest: error:", float(np.sum(y_predict != y_test) * 100) / len(y_test), "%"
    print "Gradient Boosting:\n", metrics.classification_report(y_test, y_predict), "\n"


if __name__ == "__main__":
    main()
