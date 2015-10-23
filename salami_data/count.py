#!/usr/bin/python3

import pandas
import sys

if len(sys.argv) == 1:
    print('USAGE: count.py FILE')
    sys.exit(1)
df = pandas.read_csv(sys.argv[1])
d = {}
for key in list(df['genre']):
    if key in d:
        d[key] += 1
    else:
        d[key] = 1
for k in d:
    print(d[k], k)
