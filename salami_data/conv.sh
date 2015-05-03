#!/bin/bash

if [ $# -ne 2 ]
then
    echo 'Usage: ./conv.sh basename toname'
    exit 1
fi

mkdir -p $2
for i in `seq 0 9`
do
    mv "${1}_${i}.csv.png" "${2}/${2}_${1}_${i}.png"
done
