#!/bin/bash

. /usr/bin/setwfdb

# test with all database
HOST=`hostname`
n_jobs=-1
case $HOST in
"csbb01")
    models="adaboost"
	;;
"csbb02")
    models="mlp1"
	;;
"csbb23")
    models="gradient_boosting"
    # xgboost already uses all available CPU threads so let's avoid multi-processing here.
    n_jobs=1
	;;
"csbb24")
    models="mlp2"
	;;
"csbb25")
    models="svc"
	;;
"arch-pc")
    models="random_forest logistic_regression"
	;;
esac

mkdir -p aha

iter=100
for seg_size in 8;
do
    input_features="features/features_s"$seg_size".dat"
    amendment_file="amendment_s"$seg_size".txt"
    for model in $models;
    do
        for scoring in f1_weighted;
        do
            timestamp=`date +%s`  # add current timestamp as suffix to prevent filename duplication
            output="aha/"$model","$scoring",s"$seg_size"."$timestamp".csv"
            error_log="aha/"$model","$scoring",s"$seg_size"_errors."$timestamp".csv"
            ./vf_tests.py -j $n_jobs -b -i "$input_features" -t $iter -m $model -l aha -s $scoring --amendment-file="$amendment_file" --exclude-rhythms '(ASYS' -e "$error_log" -o "$output"
        done
    done
done
