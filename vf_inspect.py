#!/usr/bin/env python3
import pyximport; pyximport.install()
import vf_features
import vf_data
import argparse
import csv
from collections import deque


MAX_QUEUE_SIZE = 50

record_cache = deque()  # a simple LRU cache for loaded records


def load_record(record_name):
    # find in the cache first (linear search is inefficient, but it works)
    record = None
    for i, cached_record in enumerate(record_cache):
        if record_name == cached_record.name:
            record = cached_record
            # move the record to end of the queue
            del record_cache[i]
            record_cache.append(record)
            break
    # not found in cache, load it
    if not record:
        record = vf_data.Record()
        names = record_name.split("/", maxsplit=1)
        db_name = names[0]
        record_id = names[1]
        channel = 1 if db_name == "mghdb" else 0
        annotator = "ari" if db_name == "mghdb" else "atr"
        record.load(db_name, record_id, channel, annotator)
        # insert into the cache
        record_cache.append(record)

        if len(record_cache) > MAX_QUEUE_SIZE:
            record_cache.popleft()  # remove one item from the cache
    return record


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-f", "--features", type=str, required=True)
    parser.add_argument("-p", "--plot", action="store_true", default=False)
    parser.add_argument("-t", "--threshold", type=float, default=0.25)
    args = parser.parse_args()

    # load features and info of the samples
    x_data, x_data_info = vf_features.load_features(args.features)

    fields = ["sample", "rhythm", "tested", "errors", "error rate"]
    errors = []
    with open(args.input, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                error_rate = float(row["error rate"])
                if error_rate > args.threshold:
                    errors.append({field: row[field] for field in fields})
            except ValueError:
                pass

    # sort the errors by error rate in descending order
    errors.sort(key=lambda item: item["error rate"], reverse=True)

    # output the errors
    for error in errors:
        print(", ".join([error[field] for field in fields]))
        sample_id = int(error["sample"])

        x_features = x_data[sample_id]
        info = x_data_info[sample_id]
        for name, feature in zip(vf_features.feature_names, x_features):
            print("{0}: {1}".format(name, feature))
        print("-" * 80)
        # plot the signals
        if args.plot:
            record = load_record(info.record_name)
            signals = record.signals[info.begin_time:info.end_time]
            vf_features.preprocessing(signals, info.sampling_rate, plotting=True)

if __name__ == '__main__':
    main()
