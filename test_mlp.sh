#!/bin/bash

# test with all database
HOST=`hostname`
case $HOST in
"csbb01")
	label=2
	;;
"csbb02")
	label=5
	;;
"csbb23")
	label=1
	;;
"csbb24")
	label=3
	;;
"csbb25")
	label=4
	;;
"arch-pc")
	label=0
	;;
esac

input_features="features/features_s8_r250.dat"
for scoring in f1 accuracy precision;
do
	output="reports/mlp_2layer_cv5_"$scoring"_s8_label"$label"_exclude_cudb.csv"
	if [ ! -f "$output" ]; then
		./vf_tests.py -b -i "$input_features" -t 20 -x -m mlp -l $label -c 5 -s $scoring -d vfdb mitdb -o "$output"
	fi

	if [ "$label" == "1" -o "$label" == "4" ]; then
		output="reports/mlp_2layer_cv5_"$scoring"_s8_label"$label"_all_db.csv"
		if [ ! -f "$output" ]; then
			./vf_tests.py -b -i "$input_features" -t 20 -x -m mlp -l $label -c 5 -s $scoring -o "$output"
		fi
	fi

	output="reports/mlp_2layer_cv5_no_weight_"$scoring"_s8_label"$label"_exclude_cudb.csv"
	if [ ! -f "$output" ]; then
		./vf_tests.py -i "$input_features" -t 20 -x -m mlp -l $label -c 5 -s $scoring -d vfdb mitdb -o "$output"
	fi
done
