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
iter=100
cv=10
for model in logistic_regression svc random_forest gradient_boosting adaboost;
do
    for scoring in ber accuracy;
    do
        output="reports/"$model"_cv10_"$scoring"_s8_label"$label"_exclude_cudb.csv"
        if [ ! -f "$output" ]; then
            ./vf_tests.py -b -i "$input_features" -t $iter -x -m $model -l $label -c $cv -s $scoring -d vfdb mitdb -o "$output"
        fi

        output="reports/"$model"_cv5_"$scoring"_s8_label"$label"_all_db.csv"
        if [ ! -f "$output" ]; then
            ./vf_tests.py -b -i "$input_features" -t $iter -x -m $model -l $label -c $cv -s $scoring -o "$output"
        fi
    done
done
