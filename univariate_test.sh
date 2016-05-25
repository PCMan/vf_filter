#!/bin/sh
mkdir -p univariate

n_iters=10
models="adaboost logistic_regression random_forest"
for model in $models;
do
    features="TCSC TCI STE MEA PSR HILB VF M A2 FM LZ SpEn MAV Count1 Count2 Count3 IMF1_LZ IMF5_LZ"
    for feature in $features;
    do
        echo "Test:" $model $feature
        ./vf_tests.py -i features/features_s8.dat -m $model -o "univariate/"$model"_s8_"$feature".csv" -t $n_iters -b -f "$feature"
    done
done
