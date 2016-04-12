#!/usr/bin/env python3
import os
import pickle

def main():
    with open("datasets/vfdb/418", "rb") as f:
        sampling_rate = pickle.load(f)
        signals = pickle.load(f)
        annotations = pickle.load(f)
        print(sampling_rate, len(signals))
        for annotation in annotations:
            print(annotation)

    return 0

if __name__ == '__main__':
    main()
