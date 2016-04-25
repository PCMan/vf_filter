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
iter=20
for scoring in ber f1 accuracy precision;
do
	# 1 hidden layer
	output="reports/mlp1_cv5_"$scoring"_s8_label"$label"_exclude_cudb.csv"
	if [ ! -f "$output" ]; then
		./vf_tests.py -b -i "$input_features" -t $iter -x -m mlp1 -l $label -c 5 -s $scoring -d vfdb mitdb -o "$output"
	fi

	if [ "$label" == "1" -o "$label" == "4" ]; then
		output="reports/mlp1_cv5_"$scoring"_s8_label"$label"_all_db.csv"
		if [ ! -f "$output" ]; then
			./vf_tests.py -b -i "$input_features" -t $iter -x -m mlp1 -l $label -c 5 -s $scoring -o "$output"
		fi
	fi

	# 2 hidden layers
	output="reports/mlp2_cv5_"$scoring"_s8_label"$label"_exclude_cudb.csv"
	if [ ! -f "$output" ]; then
		./vf_tests.py -b -i "$input_features" -t $iter -x -m mlp2 -l $label -c 5 -s $scoring -d vfdb mitdb -o "$output"
	fi

	if [ "$label" == "1" -o "$label" == "4" ]; then
		output="reports/mlp2_cv5_"$scoring"_s8_label"$label"_all_db.csv"
		if [ ! -f "$output" ]; then
			./vf_tests.py -b -i "$input_features" -t $iter -x -m mlp2 -l $label -c 5 -s $scoring -o "$output"
		fi
	fi
done
