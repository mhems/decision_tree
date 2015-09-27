#!/usr/bin/python3

from random import random as rand
import pandas as pd
import re
import sys

SALAMI_path    = '.'
val_path       = SALAMI_path + 'validations/'
prune_val_path = SALAMI_path + 'prune_validation/'
run_path       = SALAMI_path + 'runs/'

DEBUG       = False
GEN_VAL_SET = False
POST_PRUNE  = False
SAVE_GRAPH  = False

def getRelevantFeatures(dataset):
    """Drop columns of input data that are not relevant to learning"""
    return dataset.drop(['ID',
                         'std_bar_len',     'avg_bar_conf',    'std_bar_conf',
                         'std_beat_len',    'avg_beat_conf',   'std_beat_conf',
                         'std_tatum_len',   'avg_tatum_conf',  'std_tatum_conf',
                         'std_section_len', 'avg_section_conf','std_section_conf',
                         'key_val','key_conf',
                         'tempo_conf'],
                        1)

def test_tree (train_fn, test_fn, val_fn):
    """
    Learn tree on train_fn data, optionally post-prune on val_fn data
    Evaluate tree on test_fn data
    Returns number incorrect and number evaluated
    """
    dTree = DTree(train_fn, val_fn)
    df = pd.read_csv(test_fn)
    wrong = 0
    total = len(df)
    for _, row in df.iterrows():
        ID = row['ID']
        guess = dTree.decide(row)
        truth = row['genre']
        if guess != truth:
            if DEBUG:
                print(ID)
            wrong += 1
    print('File %s:: %d incorrect out of %d (%.2f%% correct)' % (
        getBaseName(test_fn),
        wrong,
        total,
        (total-wrong) * 100.0 / total))
    return (wrong, total)

def cross_validate(filepath, K):
    """
    K-fold cross-validation of files in filepath
    """
    acc_wrong = 0.0
    acc_total = 0.0
    for i in range(K):
        suffix = '_%d.csv' % i
        train_fn = filepath + 'both' + suffix
        # train_fn = filepath + 'train' + suffix
        test_fn  = filepath + 'test' + suffix
        val_fn   = None
        if POST_PRUNE:
            train_fn = filepath + 'train' + suffix
            val_fn   = filepath + 'val' + suffix
        wrong, total = test_tree(train_fn, test_fn, val_fn)
        acc_wrong += wrong
        acc_total += total
    err = acc_wrong/acc_total
    print("Total error: %.2f%%" % (err*100))
    return err

def getContiguousPartitions(lines, chunksize):
    """
    Return contiguous partition of lines where as many partitions are chunksize
    """
    chunks = [lines[start:start+chunksize] for start in range(0,
                                                              (K-1)*chunksize,
                                                              chunksize)]
    chunks.append(lines[chunksize*(K-1):])
    return chunks

def getLinesRandomly(lines, amt):
    """
    Remove amt lines randomnly from lines and return it
    """
    indices = set()
    ret = []
    while len(indices) < amt and len(lines) > 0:
        indices.add(int(rand() * len(lines)))
    for i in indices:
        ret.append(lines[i])
    indices = list(indices)
    indices.sort(reverse=True)
    for idx in indices:
        lines.pop(idx)
    return (ret, lines)

def getRandomPartitions(lines, chunksize, K):
    """
    Get K random partitions of lines of approximate size chunksize
    """
    ret = []
    i = 0
    while i < K - 1:
        t, lines = getLinesRandomly(lines, chunksize)
        ret.append(t)
        i += 1
    ret.append(lines)
    return ret

# given filename and K (num. chunks), make files
def gen_cross_validation_files(path, filename, K):
    """
    Generate K-fold cross validation files in path directory from filename
    """
    superfile = open(filename,'r')
    all_lines = superfile.readlines()
    header = all_lines[0]
    lines  = all_lines[1:]
    num_lines = len(lines)
    chunksize = num_lines/K
    # chunks = getContiguousPartitions(lines, chunksize)
    chunks = getRandomPartitions(lines, chunksize, K)
    for i, c in enumerate(chunks):
        test_f = open("%stest_%d.csv" % (path,i), 'w')
        test_f.write(header)
        test_f.writelines(c)
        r = -1
        if GEN_VAL_SET:
            r = int(rand()*K)
            while r == i:
                r = int(rand()*K)
            val_f = open('%sval_%d.csv' % (path, i), 'w')
            val_f.write(header)
            val_f.writelines(chunks[r])
            both_f = open('%sboth_%d.csv' % (path, i), 'w')
            both_f.write(header)
            both_f.writelines(chunks[r])
        rest = []
        [rest.extend(chunks[idx]) for idx in range(K) if idx != i and idx != r]
        train_f = open("%strain_%d.csv" % (path,i), 'w')
        train_f.write(header)
        train_f.writelines(rest)
        if GEN_VAL_SET:
            both_f.writelines(rest)

if __name__ == '__main__':
    K = 10
    argc = len(sys.argv)
    if argc < 2 or argc > 3:
        print('Usage: %s -m|-v|-r [-g|-f]' % sys.argv[0])
        sys.exit(1)
    if sys.argv[1] == '-m':
        if sys.argv[2] == '-p':
            GEN_VAL_SET = True
        gen_cross_validation_files("new/", SALAMI_path + 'first.csv', K)
    elif sys.argv[1] == '-v':
        if argc > 2:
            if   sys.argv[2] == '-g':
                BY_GAIN    = True
            elif sys.argv[2] == '-f':
                BY_FREQ    = True
            elif sys.argv[2] == '-b':
                BY_GAIN    = True
                BY_FREQ    = True
            elif sys.argv[2] == '-p':
                POST_PRUNE = True
        err = cross_validate(prune_val_path, K)
    # deprecated
    elif sys.argv[1] == '-r':
        print('Warning - deprecated code!')
        for i in range(1, 11):
            test_tree(run_path + 'train_' + repr(i) + '.csv',
                      run_path + 'test_'  + repr(i) + '.csv',
                      None)
            print('*' * 10)
