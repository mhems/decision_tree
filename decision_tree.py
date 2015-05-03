#!/usr/bin/python2

from random import random as rand
import pandas as pd
import re
import math
import pydot
import sys

TARGET = 'genre'
SALAMI_path = '/home/matt/Development/cs580/project/repo/salami_data/'
val_path       = SALAMI_path + 'validations/'
prune_val_path = SALAMI_path + 'prune_validation/'
run_path       = SALAMI_path + 'runs/'

BY_GAIN     = False
BY_FREQ     = False
GEN_VAL_SET = False
POST_PRUNE  = False
SAVE_GRAPH  = True
MIN_GAIN    = 0.125
MIN_FREQ    = 0.9

def getBaseName(filename):
    return re.sub('.*/', '', filename)

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

    def __init__(self, val, col, maj, l, r):
        self.value                = val
        self.col                  = col
        self.majority_class       = maj
        self.left                 = l
        self.right                = r
        self.count                = DTreeNode.i
        self.classification_error = 0
        self.num_seen             = 0
        self.prune_error          = 0
        self.pruned               = False

    @classmethod
    def makeLeaf(cls, val):
        return cls(val, -1, val, None, None)

    def decide (self, datarow):
        truth = datarow['genre']
        self.num_seen += 1
        if self.isLeaf():
            ret = self.value
        elif (datarow[self.col] < self.value):
            ret = self.left.decide(datarow)
        else:
            ret = self.right.decide(datarow)
        if ret != truth:
            self.classification_error += 1
        if self.majority_class != truth:
            self.prune_error += 1
        return ret

    def getMaxReducingNode(self):
        if self.isLeaf():
            return self
        else:
            left_n    = self.left.getMaxReducingNode()
            right_n   = self.right.getMaxReducingNode()
            left_red  = left_n.classification_error  - left_n.prune_error
            right_red = right_n.classification_error - right_n.prune_error
            my_red    = self.classification_error    - self.prune_error
            max_red   = max(left_red, right_red, my_red)
            if   max_red == left_red:
                return left_n
            elif max_red == right_red:
                return right_n
            else:
                return self

    def prune(self):
        print 'Pruning to %s, reduction: %d' % (self.majority_class, self.classification_error - self.prune_error)
        self.val    = self.majority_class
        self.col    = -1
        self.left   = None
        self.right  = None
        self.pruned = True

    def isLeaf(self):
        return self.left is None and self.right is None

    def getDescription(self):
        DTreeNode.i += 1
        diff = self.classification_error - self.prune_error
        if self.isLeaf():
            return "%s\n%f\n%d" % (self.value, diff, self.count)
        else:
            return "%s < %f\n%f\n%d" % (self.col, self.value, diff, self.count)

    def height(self):
        if self.isLeaf():
            return 0
        return max(self.left.height(), self.right.height()) + 1

    def size(self):
        if self.isLeaf():
            return 1
        return self.left.size() + self.right.size() + 1

    def toGraph(self, graph):
        myNode = self.getVertex();
        graph.add_node(myNode)
        if self.isLeaf():
            return myNode
        else:
            lNode = self.left.toGraph(graph)
            rNode = self.right.toGraph(graph)
            graph.add_node(lNode)
            graph.add_node(rNode)
            graph.add_edge(pydot.Edge(myNode, lNode))
            graph.add_edge(pydot.Edge(myNode, rNode))
            return myNode

    def getVertex(self):
        return pydot.Node(self.getDescription(), style="filled", fillcolor=self.__getColor())

    def __getColor(self):
        if self.pruned:
            return "black"
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
                color = "grey"
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

    def prettyPrint(self, indent):
        s = ' ' * indent
        if self.isLeaf():
            print s + 'return ' + repr(self.value)
        else:
            print (s + 'if %s < %f:' % (self.col, self.value))
            self.left.prettyPrint(indent + 2)
            print (s + 'else: # %s %f' % (self.col, self.value))
            self.right.prettyPrint(indent + 2)

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
    def __init__(self, filename, val_fn=None):
        dataframe = pd.read_csv(filename)
        dataframe = getRelevantFeatures(dataframe)
        print 'Learning decision tree'
        self.root = learn_decision_tree(dataframe)
        print 'Decision tree for %s has been learned' % getBaseName(filename)
        print 'Height: %d' % self.height()
        print self.size(), 'vertices'
        if val_fn is not None:
            self.post_prune(pd.read_csv(val_fn))
        print self.size(), 'vertices'
        if SAVE_GRAPH:
            self.toGraph().write_png('%s.png' % filename)

    def toGraph(self):
        graph = pydot.Dot(graph_type='digraph', ordering='out')
        self.root.toGraph(graph)
        return graph

    def height(self):
        return self.root.height()

    def size(self):
        return self.root.size()

    def decide (self, datarow):
        return self.root.decide(datarow)

    def post_prune(self, df):
        for _, row in df.iterrows():
            self.decide(row)
        node = self.root.getMaxReducingNode()
        node.prune()
        print 'Pruned height: %d' % self.height()

    def prettyPrint(self):
        self.root.prettyPrint(0)

def getMajorityClass(dataset):
    grps = dataset.groupby(TARGET).groups
    max_val = 0
    for cat in grps:
        if len(grps[cat]) > max_val:
            max_val = len(grps[cat])
            max_col = cat
    return max_col, max_val

def learn_decision_tree(dataset):
    if num_groups(dataset) == 1:
        cls = re.split('[ \t\n\r]+', repr(dataset['genre']))[1]
        DTreeNode.i += 1
        return DTreeNode.makeLeaf(cls)
    gain,axis,threshold = find_optimal_split(dataset)
    if axis == None or axis == '':
        raise Exception
    max_col, max_val = getMajorityClass(dataset)
    N = len(dataset)
    if BY_GAIN and BY_FREQ:
        if gain < MIN_GAIN and max_val * 1.0 / N > MIN_FREQ:
            return DTreeNode.makeLeaf(max_col)
    elif BY_GAIN:
        if gain < MIN_GAIN:
            return DTreeNode.makeLeaf(max_col)
    elif BY_FREQ:
        if max_val * 1.0 / N > MIN_FREQ:
            return DTreeNode.makeLeaf(max_col)
    left_subset  = dataset[dataset[axis] <  threshold]
    left_node    = learn_decision_tree(left_subset)
    right_subset = dataset[dataset[axis] >= threshold]    
    right_node   = learn_decision_tree(right_subset)
    return DTreeNode(threshold, axis, max_col, left_node, right_node)

def test_tree (train_fn, test_fn, val_fn):
    dTree = DTree(train_fn, val_fn)
    df    = pd.read_csv(test_fn)
    return evaluate(test_fn, dTree, df)

# score dTree against df truth
def evaluate(filename, dTree, df):
    wrong = 0
    total = len(df)
    for _, row in df.iterrows():
        ID = row['ID']
        guess = dTree.decide(row)
        truth = row['genre']
#        print ID, guess, truth
        if guess != truth:
            wrong += 1
    print 'File %s:: %d incorrect out of %d (%.2f%% correct)' % (getBaseName(filename), wrong, total, (total-wrong) * 100.0 / total)
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

def cross_validate(filepath, K):
    acc_wrong = 0.0
    acc_total = 0.0
    for i in range(K):
        suffix = '_%d.csv' % i
        train_fn = filepath + 'both' + suffix
        test_fn  = filepath + 'test' + suffix
        val_fn   = None
        if POST_PRUNE:
            train_fn = filepath + 'train' + suffix
            val_fn   = filepath + 'val' + suffix
        wrong, total = test_tree(train_fn, test_fn, val_fn)
        acc_wrong += wrong
        acc_total += total
    err = acc_wrong/acc_total
    print "Total error: %.2f%%" % (err*100)
    return err

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
def gen_cross_validation_files(path, filename, K):
    superfile = open(filename,'r')
    all_lines = superfile.readlines()
    header = all_lines[0]
    lines  = all_lines[1:]
    num_lines = len(lines)
    chunksize = num_lines/K
#    chunks = getContiguousPartitions(lines, chunksize)
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
        rest = []
        [rest.extend(chunks[idx]) for idx in range(K) if idx != i and idx != r]
        train_f = open("%strain_%d.csv" % (path,i), 'w')
        train_f.write(header)
        train_f.writelines(rest)

if __name__ == '__main__':
    K = 10
    argc = len(sys.argv)
    if argc < 2 or argc > 3:
        print 'Usage: %s -m|-v|-r [-g|-f]' % sys.argv[0]
        sys.exit(1)
    if sys.argv[1] == '-m':
        if sys.argv[2] == '-p':
            GEN_VAL_SET = True
        gen_cross_validation_files("new/", SALAMI_path + 'first.csv', K)
    elif sys.argv[1] == '-v':
        if argc > 2:
            if sys.argv[2] == '-g':
                BY_GAIN = True
            elif sys.argv[2] == '-f':
                BY_FREQ = True
            elif sys.argv[2] == '-b':
                BY_GAIN = True                
                BY_FREQ = True
            elif sys.argv[2] == '-p':
                POST_PRUNE = True
        err = cross_validate(prune_val_path, K)
    elif sys.argv[1] == '-r':
        for i in range(1,11):
            test_tree(run_path + 'train_' + repr(i) + '.csv',
                      run_path + 'test_'  + repr(i) + '.csv')
            print '**********'
