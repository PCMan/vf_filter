#!/usr/bin/env python3
import pyximport; pyximport.install()
import vf_data
import argparse


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
(VFL		Ventricular flutter
(VT	Ventricular tachycardia
"""
rhythm_names = {}
for line in rhythm_name_tab.split("\n"):
    cols = line.split("\t", maxsplit=1)
    if len(cols) > 1:
        name = cols[0].strip()
        desc = cols[1].strip()
        rhythm_names[name] = desc


def main():
    all_db_names = ("mitdb", "vfdb", "cudb")
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--db-names", type=str, nargs="+", choices=all_db_names, default=all_db_names)
    # parser.add_argument("-o", "--output", type=str)
    parser.add_argument("-x", "--exclude-noise", action="store_true", default=False)
    parser.add_argument("-s", "--segment-duration", type=int, default=8)
    args = parser.parse_args()

    n_segments = 0
    all_rhythm_statistics = {}
    for db_name in args.db_names:
        print(db_name)
        rhythm_statistics = {}
        for record_name in vf_data.get_records(db_name):
            # load the record from the ECG database
            record = vf_data.Record()
            record.load(db_name, record_name)
            for segment in record.get_segments(args.segment_duration):
                info = segment.info
                if args.exclude_noise and info.has_artifact:
                    continue
                # print(info.record, info.begin_time, info.rhythm_types)
                for rhythm_name in info.rhythm_types:
                    rhythm_statistics[rhythm_name] = rhythm_statistics.get(rhythm_name, 0) + 1
                n_segments += 1

        for name in sorted(rhythm_statistics.keys()):
            desc = rhythm_names.get(name, name)
            n = rhythm_statistics.get(name)
            print(name, "\t", n, "\t", desc)
            all_rhythm_statistics[name] = all_rhythm_statistics.get(name, 0) + n
        print("-" * 80)

    print("Summary:")
    n_rhythms = 0
    for name in sorted(all_rhythm_statistics.keys()):
        desc = rhythm_names.get(name, name)
        n = all_rhythm_statistics.get(name)
        print(name, "\t", n, "\t", desc)
        n_rhythms += n
    print("\t", n_segments - n_rhythms, "\tUnknown rhythm")


if __name__ == "__main__":
    main()
