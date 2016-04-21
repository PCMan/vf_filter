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
	label=1
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
	./vf_tests.py -b -i "$input_features" -t 20 -x -m mlp -l $label -c 5 -s $scoring -d vfdb mitdb -o "reports/mlp_2layer_cv5_"$scoring"_s8_label"$label"_exclude_cudb.csv"
	if [ "$label" == "1" -o "$label" == "4" ]; then
		./vf_tests.py -b -i "$input_features" -t 20 -x -m mlp -l $label -c 5 -s $scoring -o "reports/mlp_2layer_cv5_"$scoring"_s8_label"$label"_all_db.csv"
	fi
done
