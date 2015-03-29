#!/bin/bash

sed -i 's@Alternative Pop / @@g' merge.csv
#sed -i 's/Autres/Rock/g' merge.csv

grep -E 'Classical|World|Rock|Blues|R&B|Jazz' merge.csv
