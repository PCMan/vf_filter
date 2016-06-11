#!/usr/bin/env python3
import csv
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, nargs="+", required=True)
    parser.add_argument("-o", "--output", type=str, required=True)
    args = parser.parse_args()

    header = ["sample", "record", "begin", "rhythm"]
    header2 = ["tested", "errors", "error rate", "class", "predict"]

    all_sample_infos = []
    all_results = []
    n_tests = 0
    for filename in args.input:
        with open(filename, "r", newline='') as f:
            reader = csv.reader(f)
            field_names = next(reader)  # read header
            n_tests += len(field_names) - (len(header) + len(header2))
            for i, row in enumerate(reader):
                if i < len(all_sample_infos):  # we already know about the sample
                    all_results[i].extend(row[len(header):-len(header2)])
                else:  # build an entry for the sample
                    sample_info = row[0:len(header)]
                    sample_info.append(row[-2])  # -1 is actual class
                    all_sample_infos.append(sample_info)
                    all_results.append(row[len(header):-len(header2)])

    with open(args.output, "w", newline='') as f:
        writer = csv.writer(f)
        fields = header + list(range(1, n_tests + 1)) + header2
        writer.writerow(fields)  # write csv header
        for sample_info, sample_results in zip(all_sample_infos, all_results):
            # calculate statistics
            actual_class = sample_info.pop(-1)  # remove actual class
            sample_id = i +1
            predicted = {}
            n_tested = 0
            n_errors = 0
            for y in sample_results:
                if y != "":  # selected in this test iteration
                    n_tested += 1
                    n = predicted.get(y, 0)
                    predicted[y] = n + 1
                    if y != actual_class:
                        n_errors += 1
            # find the mostly predicted class
            n_class = 0
            for c, n in predicted.items():
                if n > n_class:
                    most_frequent_predict = c
            row = []
            row.extend(sample_info)
            row.extend(sample_results)
            row.extend([n_tested, n_errors, n_errors/n_tested if n_tested else "N/A", actual_class, most_frequent_predict])
            writer.writerow(row)


if __name__ == '__main__':
    main()
