#!/usr/bin/env python3
import pyximport; pyximport.install()  # use Cython
import qrs_detect
import vf_data
import signal_processing
import argparse


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--update-labels", type=str)  # try to override incorrect labels
    args = parser.parse_args()

    dataset = vf_data.DataSet()

    # load existing correction file
    # manually corrections for incorrect labels coming from annotations of the original dataset
    corrections = {}
    corrections_changed = False
    if args.update_labels:  # if we're going to update incorrect labels
        dataset.load_correction(args.update_labels)
        # read the label correction file
        with open(args.update_labels, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 3:
                    continue
                record_name, begin_time, correct_label = parts
                corrections[(record_name, int(begin_time))] = correct_label

    for segment in dataset.get_samples(8):
        info = segment.info
        rhythm = info.rhythm
        samples = segment.signals
        # trend removal (drift suppression)
        samples = signal_processing.drift_supression(samples, 1, info.sampling_rate)
        # smoothing
        samples = signal_processing.moving_average(samples)

        amplitude = signal_processing.get_amplitude(samples, info.sampling_rate)
        # beats = qrs_detect.qrs_detect(signals, info.sampling_rate, info.adc_zero, info.gain)

        # if the amplitude is very low but the label is not asystole, change it
        if amplitude < 0.15 and rhythm != "(ASYS":
            print(info.record_name, info.begin_time, rhythm, amplitude, " ==> change the label to asystole")
            corrections[(info.record_name, info.begin_time)] = "(ASYS"
            corrections_changed = True

    if args.update_labels and corrections_changed:  # if we're going to update incorrect labels
        # save the updated label corrections back to file
        with open(args.update_labels, "w") as f:
            for key in sorted(corrections.keys(), key=lambda k: "{0}:{1}".format(k[0], k[1])):
                correct_label = corrections[key]
                f.write("{0}\t{1}\t{2}\n".format(key[0], key[1], correct_label))


if __name__ == '__main__':
    main()
