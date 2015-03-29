import re
import pandas
import sys
import numpy

f = sys.argv[1]
df = pandas.read_csv(f)
#df = df.sort('genre')
d = {}
for key in list(df['genre']):
#    key = re.sub('-.*$','',key)
#    key = re.sub("u'",'',key)
    if key in d:
        d[key] += 1
    else:
        d[key] = 1
for k in d:
    print d[k],k
