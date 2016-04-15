#!/bin/sh
mkdir -p univariate_rf
for i in $*; do
    ./vf_tests.py --model random_forest --output "univariate_rf/feature_"$i".csv" --scorer=ber -t 10 --cv-fold 10 --features $i
done
