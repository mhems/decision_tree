#!/usr/bin/python2

from random import random as rand
import pandas as pd
import numpy as np
import math
import re
import glob
import math
import pydot
import sys

TARGET = 'genre'
SALAMI_path = '/home/matt/Development/cs580/project/repo/salami_data/'
val_path = SALAMI_path + 'rand_validation/'
run_path = SALAMI_path + 'runs/'

BY_GAIN    = False
BY_FREQ    = False
SAVE_GRAPH = False

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
    i = 0
    def __init__(self,val,col=-1,l=None,r=None):
        self.value = val
        self.col = col
        self.left = l
        self.right = r
        self.count = DTreeNode.i

    def decide (self, datarow):
        if self.isLeaf():
            return self.value
        if (datarow[self.col] < self.value):
            return self.left
        return self.right

    def isLeaf(self):
        return self.left is None and self.right is None

    def getDescription(self):
        DTreeNode.i += 1
        if self.isLeaf():
            return "%s\n%d" % (self.value, self.count)
        else:
            return "%s < %d\n%d" % (self.col, self.value, self.count)

    def getVertex(self):
        return pydot.Node(self.getDescription(), style="filled", fillcolor=self.__getColor());

    def __getColor(self):
        if (self.isLeaf()):
            if self.value == "Blues":
                color = "blue"
            elif self.value == "Classical":
                color = "orange"
            elif self.value == "Jazz":
                color = "red"
            elif self.value == "R&B":
                color = "yellow"
            elif self.value == "Rock":
                color = "green"
            elif self.value == "World":
                color = "pink"
        else:
            if self.col == "num_bars" or self.col == "avg_bar_len":
                color = "black"
            elif self.col == "num_beats" or self.col == "avg_beat_len":
                color = "white"
            elif self.col == "num_tatums" or self.col == "avg_tatum_len":
                color = "khaki"
            elif self.col == "num_sections" or self.col == "avg_section_len":
                color = "orchid"
            elif self.col == "tempo_val":
                color = "lavender"
            elif self.col == "duration":
                color = "brown"
        return color

def num_groups(dataset):
    return len(dataset.groupby(TARGET).groups)

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
        graph = pydot.Dot(graph_type='digraph')
        self.root = learn_decision_tree(dataframe, graph)
        print 'Decision tree for %s has been learned' % filename
        print 'Max. depth: %d' % self.length()
        if SAVE_GRAPH:
            graph.write_png('%s.png' % filename)
        
    def length(self):
        return self.__len(self.root)

    def __len(self, node):
        if node.isLeaf():
            return 0
        return max(self.__len(node.left), self.__len(node.right)) + 1

    def decide (self, datarow):
        return self.decide_rec(self.root, datarow)

    def decide_rec(self, node, datarow):
        dec = node.decide(datarow)
        if type(dec) == type(""):
            return dec
        return self.decide_rec(dec, datarow)

    def prettyPrint(self):
        self.rec_print(self.root,"")

    def rec_print(self, node, indent):
        s = ' ' * indent
        if (node.left != None):
            print (s + 'if %s < %f:' % (node.col, node.value))
            self.rec_print(node.left, indent + 2)
            print (s + 'else: # %s %f' % (node.col, node.value))
            self.rec_print(node.right, indent + 2)
        else:
            print s + 'return ' + repr(node.value)

def learn_decision_tree(dataset, graph):
    if num_groups(dataset) == 1:
        cls = re.split('[ \t\n\r]+', repr(dataset['genre']))[1]
        #print("  LEAF: '%s'" % (cls))
        DTreeNode.i += 1
        return DTreeNode(cls)
    #print("  INTERNAL %d %d" % (num_groups(dataset),len(dataset)))
    gain,axis,threshold = find_optimal_split(dataset)
    if axis == None or axis == '':
        raise Exception
    if BY_GAIN:
        if gain < 0.1375:
            grps = dataset.groupby(TARGET).groups
            max_val = 0
            for cat in grps:
                if len(grps[cat]) > max_val:
                    max_val = len(grps[cat])
                    max_col = cat
            return DTreeNode(max_col)
    elif BY_FREQ:
        grps = dataset.groupby(TARGET).groups
        N = len(dataset)
        max_val = 0
        for cat in grps:
            if len(grps[cat]) > max_val:
                max_val = len(grps[cat])
                max_col = cat
        print max_val * 1.0 / N
        if max_val * 1.0 / N > 0.9:
            return DTreeNode(max_col)
        # if one group comprises sufficient percent of total, return that class

    left_subset  = dataset[dataset[axis] <  threshold]
    left_node    = learn_decision_tree(left_subset, graph)
    right_subset = dataset[dataset[axis] >= threshold]    
    right_node   = learn_decision_tree(right_subset, graph)

    left_vertex  = left_node.getVertex()
    right_vertex = right_node.getVertex()
    graph.add_node(left_vertex)
    graph.add_node(right_vertex)
    parent = DTreeNode(threshold, axis, left_node, right_node)
    parent_vertex = parent.getVertex()
    graph.add_node(parent_vertex)
    graph.add_edge(pydot.Edge(parent_vertex, left_vertex))
    graph.add_edge(pydot.Edge(parent_vertex, right_vertex))
    return parent

def test_tree (train_fn,test_fn):
    dTree = DTree(train_fn)
    df    = pd.read_csv(test_fn)
    wrong = 0
    total = len(df)
    for _,row in df.iterrows():
        ID = row['ID']
        guess = dTree.decide(row)
        truth = row['genre']
#        print ID, guess, truth
        if guess != truth:
            wrong += 1
    print 'File %s:: %d incorrect out of %d (%.2f%% correct)' % (test_fn,wrong,total, (total-wrong) * 100.0 / total)
    return (wrong,total)

def getContiguousPartitions(lines, chunksize):
    chunks = [lines[start:start+chunksize] for start in range(0,(K-1)*chunksize, chunksize)]
    chunks.append(lines[chunksize*(K-1):])
    return chunks

# return removed amt lines from lines
def getLinesRandomly(lines, amt):
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
    ret = []
    i = 0
    while i < K - 1:
        t, lines = getLinesRandomly(lines, chunksize)
        ret.append(t)
        i += 1
    ret.append(lines)
    return ret

# given filename and K (num. chunks), make files
def gen_cross_validation_files(filename, K):
    superfile = open(filename,'r')
    all_lines = superfile.readlines()
    header = all_lines[0]
    lines  = all_lines[1:]
    num_lines = len(lines)
    chunksize = num_lines/K
#    chunks = getContiguousPartitions(lines, chunksize)
    chunks = getRandomPartitions(lines, chunksize, K)
#    print chunks
#    return
    for i, c in enumerate(chunks):
        test_f = open("%stest_%d.csv" % (val_path,i), 'w')
        test_f.write(header)
        test_f.writelines(c)
        rest = []
        [rest.extend(r) for r in chunks[0:i]]
        [rest.extend(r) for r in chunks[i+1:]]
        train_f = open("%strain_%d.csv" % (val_path,i), 'w')
        train_f.write(header)
        train_f.writelines(rest)

def cross_validate(K):
    acc_wrong = 0.0
    acc_total = 0.0
    for i in range(0,K):
        wrong, total = test_tree('%strain_%d.csv' % (val_path,i), '%stest_%d.csv' % (val_path,i))
        acc_wrong += wrong
        acc_total += total
    err = acc_wrong/acc_total
    print "Total error: %.2f%%" % (err*100)
    return err

if __name__ == '__main__':
    argc = len(sys.argv)
    if argc < 2 or argc > 3:
        print 'Usage: %s -m|-v|-r [-g|-f]' % sys.argv[0]
        sys.exit(1)
    if sys.argv[1] == '-m':
        gen_cross_validation_files(SALAMI_path + 'first.csv', 10)
    elif sys.argv[1] == '-v':
        if argc > 2:
            if sys.argv[2] == '-g':
                BY_GAIN = True
            elif sys.argv[2] == 'f':
                BY_FREQ = True
        err = cross_validate(10)
    elif sys.argv[1] == '-r':
        for i in range(1,11):
            test_tree(run_path + 'train_' + repr(i) + '.csv',
                      run_path + 'test_'  + repr(i) + '.csv')
            print '*' * 10

