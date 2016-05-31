#!/usr/bin/env python3
import pyximport; pyximport.install()
import csv
import argparse
from vf_features import feature_names
import numpy as np


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True)
    args = parser.parse_args()

    with open(args.input, "r") as f:
        n_features = len(feature_names)  # FIXME: we should only use selected features
        n_classes = 3
        results = np.empty(shape=[n_classes, n_features], dtype="float64")
        reader = csv.DictReader(f)
        n_rows = 0
        for row in reader:
            for class_id in range(0, n_classes):
                # calculate coefficients for this classification problem
                coef = np.zeros(n_features)
                for i, feature in enumerate(feature_names):
                    field = "{0}[{1}]".format(feature, class_id)
                    coef[i] = row[field]
                # normalize coefficients
                coef /= np.linalg.norm(coef)
                # print("test", row["iter"], coef)
                results[class_id, :] += coef
            n_rows += 1

        # average for all iterations
        for class_id in range(0, n_classes):
            results[class_id] /= n_rows

        print("Feature importance estimated from regression coefficients:")
        for class_id in range(0, n_classes):
            print("class", class_id)
            print("-" * 80)
            class_results = results[class_id]
            sorted_idx = np.argsort(np.abs(class_results))
            for i in reversed(sorted_idx):
                print(feature_names[i], class_results[i])
            print("-" * 80)

if __name__ == "__main__":
    main()
