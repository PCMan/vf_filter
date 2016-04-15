#!/bin/sh
for i in 0 1 2 3 4 5 6 7 8 9; do
    ./vf_tests.py --model svc --output /dev/null  --cv-fold 10 --features $i
done
