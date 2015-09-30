#!/usr/bin/python3

from random import random as rand
import pandas as pd
import argparse

from Utilities import getRandomPartitions, getContiguousPartitions

def getDataFrame(filename):
    """Parse filename into dataframe"""
    dataset = pd.read_csv(filename)
    # Drop columns of input data that are not relevant to learning
    return dataset.drop(['ID',
                         'std_bar_len',     'avg_bar_conf',    'std_bar_conf',
                         'std_beat_len',    'avg_beat_conf',   'std_beat_conf',
                         'std_tatum_len',   'avg_tatum_conf',  'std_tatum_conf',
                         'std_section_len', 'avg_section_conf','std_section_conf',
                         'key_val', 'key_conf',
                         'tempo_conf'],
                        1)

def __test_tree (train_fn, test_fn, val_fn=None):
    """
    Learn tree on train_fn data, optionally post-prune on val_fn data
    Evaluate tree on test_fn data
    Returns number incorrect and number evaluated
    """
    dTree = DTree.learn_decision_tree(getDataFrame(train_fn))
    if val_fn is not None:
        dTree.post_prune(getDataFrame(val_fn))
    df = getDataFrame(test_fn)
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
    """K-fold cross-validation of files in filepath"""
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
        wrong, total = __test_tree(train_fn, test_fn, val_fn)
        acc_wrong += wrong
        acc_total += total
    err = acc_wrong/acc_total
    print("Total error: %.2f%%" % (err*100))
    return err

def gen_cross_validation_files(path,
                               filename,
                               K,
                               partition_func,
                               generate_for_prune=False):
    """
    Generate K-fold cross validation files in path directory from filename
    """
    superfile = open(filename,'r')
    all_lines = superfile.readlines()
    header = all_lines[0]
    lines  = all_lines[1:]
    N = len(lines)

    partitions = partition_func(lines, K)
    for i, c in enumerate(partitions):
        test_f = open("%stest_%d.csv" % (path,i), 'w')
        test_f.write(header)
        test_f.writelines(c)
        r = -1
        if generate_for_prune:
            r = int(rand()*K)
            while r == i:
                r = int(rand()*K)
            val_f = open('%sval_%d.csv' % (path, i), 'w')
            val_f.write(header)
            val_f.writelines(partitions[r])
            both_f = open('%sboth_%d.csv' % (path, i), 'w')
            both_f.write(header)
            both_f.writelines(partitions[r])
        rest = []
        for idx in range(K):
            if idx != i and idx != r:
                rest.extend(partitions[idx])
        train_f = open("%strain_%d.csv" % (path,i), 'w')
        train_f.write(header)
        train_f.writelines(rest)
        if generate_for_prune:
            both_f.writelines(rest)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Music Genre Classification using decision trees")
    parser.add_argument('-p',
                        dest = 'val_file',
                        metavar = 'FILE',
                        default = None,
                        help = 'validation file for post-pruning decision tree')
    parser.add_argument('-d',
                        action = 'store_true',
                        default = False,
                        help = 'print debugging information')
    parser.add_argument('-k',
                        default = 10,
                        metavar = 'K',
                        type = int,
                        help = 'K-fold cross-validation')
    parser.add_argument('-s',
                        dest = 'graph_file',
                        metavar = 'FILE',
                        help = 'save tree as image to file')
    parser.add_argument('-c',
                        choices = ['g', 'f', 'b'],
                        default = 'g',
                        dest = 'validation_method',
                        metavar = 'METHOD',
                        help = 'cross-validation method, one of (g)ain,'
                               '(f)req, or (b)oth')
    parser.add_argument('-g',
                        choices = ['r', 'c'],
                        default = 'r',
                        dest = 'generation_method',
                        metavar = 'METHOD',
                        help = 'generate files for cross-validation either'
                               '(r)andomly or (c)ontiguously')
    parser.add_argument('-L',
                        metavar = 'LOCATION',
                        help = 'directory to put generated files under')
    parser.add_argument('FILE',
                        help = 'csv file to load data from')

    args = parser.parse_args()

    if args.g is not None:
        if args.generation_method == 'r':
            func = Utilities.getRandomPartitions
        elif args.generation_method == 'c':
            func = Utilities.getContiguousPartitions
        gen_cross_validation_files(args.L, args.FILE, args.K, func, args.p)

    dataset = getDataFrame(args.FILE)
    if   args.validation_method == 'g':
        param = DTree.BY_GAIN
    elif args.validation_method == 'f':
        param = DTree.BY_FREQ
    elif args.validation_method == 'b':
        param = DTree.BY_BOTH
        dTree = DTree.learn_decision_tree(dataset, method=param)

    if args.p:
        val_data = getDataFrame(args.val_file)
        dTree.post_prune(val_data)

    if args.c is not None:
        cross_validate(dTree, args.L, args.K)

    if args.s is not None:
        dTree.writeGraph(args.graph_file)
