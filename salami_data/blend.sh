#!/bin/bash

for i in `seq 0 9`
do
    cat train_$i.csv val_$i.csv > both_$i.csv
done
