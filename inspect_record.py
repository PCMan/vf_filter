#!/usr/bin/env python3
import pyximport; pyximport.install()
import vf_data
import sys


def main(argv):
    if len(argv) < 3:
        return
    record = vf_data.Record()
    record.load(argv[1], argv[2])
    for rhythm in record.get_artifact_free_rhythms():
        print(rhythm.name, rhythm.begin_time, rhythm.end_time, rhythm.begin_beat, rhythm.end_beat)

if __name__ == '__main__':
    main(sys.argv)
