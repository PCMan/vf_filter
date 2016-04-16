#!/bin/sh
mkdir -p univariate_svm
for i in $*; do
    ./vf_tests.py --model svc --output "univariate_svm/feature_"$i".csv" -t 100 -b --cv-fold 10 --features $i
done
