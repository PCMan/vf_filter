#!/bin/sh
mkdir -p univariate

n_iters=10
models="adaboost logistic_regression random_forest"
for model in $models;
do
    features="TCSC TCI STE MEA PSR HILB VF M A2 FM LZ SpEn MAV C1 C2 C3"
    for feature in $features;
    do
        echo "Test:" $model $feature
        ./vf_tests.py -i features/features_s8.dat -m $model -o "univariate/"$model"_s8_"$feature".csv" -t $n_iters -b -f "$feature"
    done
done
