#!/bin/bash

. /usr/bin/setwfdb

# test with all database
HOST=`hostname`
case $HOST in
"csbb01")
    models="adaboost"
	;;
"csbb02")
    models="random_forest"
	;;
"csbb23")
    models="gradient_boosting"
	;;
"csbb24")
    models="mlp2"
	;;
"csbb25")
    models="svc mlp1"
	;;
"arch-pc")
    models="logistic_regression"
	;;
esac

mkdir -p aha

iter=100
for seg_size in 5 8 10;
do
    input_features="features/features_s"$seg_size"_r250.dat"
    for model in $models;
    do
        for scoring in f1_weighted accuracy;
        do
            timestamp=`date +%s`  # add current timestamp as suffix to prevent filename duplication
            output="aha/"$model","$scoring",s8."$timestamp".csv"
            error_log="aha/"$model","$scoring",s"$seg_size"_errors."$timestamp".csv"
            ./vf_tests.py -b -i "$input_features" -t $iter -m $model -l aha -s $scoring -e "$error_log" -o "$output"
        done
    done
done
