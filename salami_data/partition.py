#!/usr/bin/python2

####################
# USAGE python partition.py filename percent_test
####################

import sys
from random import random as rand

# returns (train lines, test lines)
def partition(lines, percent):
    num = len(lines)
    test_no = (num * percent) // 100
    train_no  = num - test_no
    train = lines
    test = []
    s = set()
    while len(s) < test_no:
        s.add(int(rand() * (train_no - len(s) - 1)))
    for e in s:
        test.append(train[e])
    for e in s:
        train.pop(e)
    return (train,test)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print 'USAGE: python partition.py filename percent_test'
        sys.exit(1)
    percent_test = int(sys.argv[2])
    if percent_test < 0 or percent_test > 100:
        print 'invalid percent'
        sys.exit(1)
    filenam = sys.argv[1]
    train_n = 'train_' + filenam
    test_n  = 'test_'  + filenam
    par_f   = open(filenam)
    train_f = open(train_n, 'w')
    test_f  = open(test_n, 'w')
    lines  = par_f.readlines()
    header = lines[0]
    train,test = partition(lines[1:], percent_test)
    train_f.write(header)
    train_f.writelines(train)
    test_f.write(header)
    test_f.writelines(test)
    print "Files %s (%d%%) %s (%d%%) created" % (train_n,100-percent_test,test_n,percent_test)
