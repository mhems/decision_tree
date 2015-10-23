#!/bin/bash

if [[ $# -ne 1 ]]
then
    echo 'USAGE: merge.sh FILE'
    exit 1
fi

sed -ri 's@,([^,]+)$@,"\1"@g' $1
sed -ri 's/^([^,]+),/"\1",/g' $1

sed -i 's@Alternative Pop / @@g' $1
#sed -i 's/Autres/Rock/g' $1

grep -E 'Classical|World|Rock|Blues|R&B|Jazz' $1
