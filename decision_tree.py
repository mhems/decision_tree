#!/usr/bin/python2


import pandas as pd
import numpy as np
import math
import re
import glob
import math

TARGET = 'genre'

def entropy(array):
    freq_dict = {}
    for val in array:
        if val not in list(freq_dict):
            freq_dict[val]  = 1
        else:
            freq_dict[val] += 1
    entropy = 0.0
    total_count = 1.0 * len(array)
    for val in list(freq_dict):
        frequency = 1.0 * freq_dict[val] / total_count
        entropy  -= frequency * math.log(frequency,2)
    return entropy

def information_gain(dataset, axis, threshold):
    y = list(dataset[TARGET])
    ye = entropy(y)
    a = list(dataset[dataset[axis] <  threshold][TARGET])
    ae = (1.0 * len(a)/len(y)) * entropy(a)
    b = list(dataset[dataset[axis] >= threshold][TARGET])
    be = (1.0 * len(b)/len(y)) * entropy(b)
    return ye - ae - be

def find_optimal_split(dataset):
    (best_gain,best_axis,best_threshold) = (0,0,0)
    axis_index = 0
    features = dataset.columns.tolist()[:-1]
    for axis in features:
        uniq_data = dataset.sort(columns=axis,inplace=False)
        uniq_data = uniq_data.drop_duplicates(subset=axis)
        for index in range(0,len(uniq_data) - 1):
            datum1 = uniq_data.iloc[index,axis_index]
            datum2 = uniq_data.iloc[index + 1, axis_index]
            threshold = datum1 + (abs(datum1 - datum2)/2)
            gain = information_gain(dataset,axis,threshold)
            if (gain > best_gain):
                (best_gain,best_axis,best_threshold) = (gain,axis,threshold)
        axis_index += 1
    group_dict = dataset.groupby(TARGET).groups
    return best_gain, best_axis, best_threshold
    
class DTreeNode:
    def __init__(self,val,col=-1,l=None,r=None):
        self.value = val
        self.col = col
        self.left = l
        self.right = r

    def decide (self, datarow):
        if self.isLeaf():
            return self.value
        if (datarow[self.col] < self.value):
            return self.left
        return self.right

    def isLeaf(self):
        return self.left is None and self.right is None

def num_groups(dataset):
    return len(dataset.groupby(TARGET).groups)

def learn_decision_tree(dataset):
    if num_groups(dataset) == 1:
        cls = re.split('[ \t\n\r]+',repr(dataset['genre']))[1]
        #print("  LEAF: '%s'" % (cls))
        return DTreeNode(cls)
    #print("  INTERNAL %d %d" % (num_groups(dataset),len(dataset)))
    _,axis,threshold = find_optimal_split(dataset)
    if axis == None or axis == '':
        raise Exception
    left_subset  = dataset[dataset[axis] <  threshold]
    right_subset = dataset[dataset[axis] >= threshold]    
    return DTreeNode(threshold,
                     axis,
                     learn_decision_tree(left_subset),
                     learn_decision_tree(right_subset))


def getRelevantFeatures(dataset):
    # strip away columns not used for learning
    return dataset.drop(['ID',
                         'std_bar_len','avg_bar_conf','std_bar_conf',
                         'std_beat_len', 'avg_beat_conf','std_beat_conf',
                         'std_tatum_len', 'avg_tatum_conf','std_tatum_conf',
                         'std_section_len', 'avg_section_conf','std_section_conf',
                         'key_val','key_conf','tempo_conf'
                       ], 1)

class DTree:
    def __init__(self, filename):
        dataframe = pd.read_csv(filename)
        dataframe = getRelevantFeatures(dataframe)
        print 'Learning decision tree'
        self.root = learn_decision_tree(dataframe)
        print 'Decision tree for %s has been learned' % filename

    def decide (self, datarow):
        return self.decide_rec(self.root, datarow)

    def decide_rec(self, node, datarow):
        dec = node.decide(datarow)
        if type(dec) == type(""):
            return dec
        return self.decide_rec(dec, datarow)

def test_tree (train_fn,test_fn):
    dTree = DTree(train_fn)
    df    = pd.read_csv(test_fn)
    wrong = 0
    total = len(df)
    for _,row in df.iterrows():
        ID = row['ID']
        guess = dTree.decide(row)
        truth = row['genre']
        print ID, guess, truth
        if guess != truth:
            wrong += 1
    print 'File %s::%d incorrect out of %d (%.2f%%)' % (train_fn,wrong,total,wrong * 100 / total)

if __name__ == '__main__':
    SALAMI_path = '/home/matt/Development/cs580/project/repo/salami_data/runs/'
    for i in range(1,11):
        test_tree(SALAMI_path + 'train_' + repr(i) + '.csv',
                  SALAMI_path + 'test_'  + repr(i) + '.csv')
        print '*' * 10

def rec_print(node, indent):
    s = ' ' * indent
    if (node.left != None):
        print (s + 'if %s < %f:' % (node.col, node.value))
        rec_print(node.left, indent + 2)
        print (s + 'else:' % (node.col, node.value))
        rec_print(node.right, indent + 2)
    else:
        print s + 'return ' + repr(node.value)
