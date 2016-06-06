#!/usr/bin/env python3
import csv
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, nargs="+", required=True)
    parser.add_argument("-o", "--output", type=str, required=True)
    args = parser.parse_args()

    header = None
    rows = []
    for filename in args.input:
        with open(filename, "r", newline='') as f:
            reader = csv.DictReader(f)
            if not header:
                header = reader.fieldnames
            for row in reader:
                if row["iter"] == "average":
                    break
                row["iter"] = len(rows) + 1
                rows.append(row)

    # calculate_average
    avg = {"iter": "average"}
    for field in header[1:]:
        col = []
        for row in rows:
            try:
                value = float(row.get(field, 0.0))
            except ValueError:
                value = 0.0
            col.append(value)
        mean = np.mean(col)
        if field.startswith("Se") or field.startswith("Sp") or field.startswith("precision") or field.startswith("AHA_"):
            mean = "{0:.2f}%".format(mean * 100)
        avg[field] = mean
    rows.append(avg)

    with open(args.output, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
