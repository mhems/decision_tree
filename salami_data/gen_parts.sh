#!/bin/bash

for i in `seq 1 10`
do
    ./partition.py first.csv 10
    mv train_first.csv train_$i.csv
    mv test_first.csv  test_$i.csv
done
