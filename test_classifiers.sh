#!/bin/bash

. /usr/bin/setwfdb

# test with all database
HOST=`hostname`
n_jobs=-1
case $HOST in
"csbb01")
    models="svc_linear svc_poly"
	;;
"csbb02")
    models="adaboost"
    ;;
"csbb03")
    models="random_forest"
	;;
"csbb04")
    models="mlp1"
    ;;
"csbb05")
    models="mlp2"
    ;;
"csbb06")
    models="gradient_boosting"
    # xgboost already uses all available CPU threads so let's avoid multi-processing here.
    # n_jobs=1
	;;
#"csbb23")
#    models="mlp2"
#	;;
#"csbb23")
#    models="mlp2"
#	;;
"csbb25")
    models="svc_rbf"
	;;
"arch-pc")
    models="logistic_regression"
	;;
esac

mkdir -p aha

iter=100
for seg_size in 8;
do
    input_features="features/features_s"$seg_size".dat"
    for model in $models;
    do
        for scoring in custom f1_weighted;
        do
            timestamp=`date +%s`  # add current timestamp as suffix to prevent filename duplication
            file_name="aha/"$model","$scoring",s"$seg_size"."$timestamp
            output=$file_name".csv"
            error_log=$file_name"_errors.csv"
            ./vf_tests.py -i "$input_features" -t $iter -m $model -s $scoring -e "$error_log" -o "$output"
        done
    done
done
