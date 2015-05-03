#!/bin/bash

for i in `seq 0 9`
do
    grep -v 'ID' val_$i.csv > tmp
    cat train_$i.csv tmp > both_$i.csv
done
rm -f tmp
