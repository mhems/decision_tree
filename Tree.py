import pydot
import re

from Utilities import findOptimalSplit, num_groups, getMajorityClass

DEBUG = False

genre_map = {
    "Blues":     "blue",
    "Classical": "orange",
    "Jazz":      "red",
    "R&B":       "yellow",
    "Rock":      "green",
    "World":     "pink"
}

feature_map = {
    "num_bars" :       "sienna",
    "avg_bar_len":     "sienna",
    "num_beats" :      "white",
    "avg_beat_len":    "white",
    "num_tatums" :     "khaki",
    "avg_tatum_len":   "khaki",
    "num_sections" :   "orchid",
    "avg_section_len": "orchid",
    "tempo_val":       "lavender",
    "duration":        "brown"
}

class BinaryTree:
    """Class representing full binary tree"""

    def __init__(self, node, left=None, right=None):
        self.node  = node
        self.left  = left
        self.right = right

    @property
    def node(self):
        """Return data encapsulated in node"""
        return self.node

    @property
    def isLeaf(self):
        """Return True iff tree has no children"""
        return self.left is None and self.right is None

    @property
    def height(self):
        """Return 1-based height of tree"""
        if self.isLeaf:
            return 1
        else:
            return 1 + max(self.left.height, self.right.height)

    @property
    def size(self):
        """Return number of nodes in tree"""
        if self.isLeaf:
            return 1
        else:
            return 1 + self.left.size + self.right.size

class DTreeData:
    """Encapsulate data for a node within a decision tree"""
    i = 0

    def __init__(self, val, col, maj):
        self.value                = val
        self.col                  = col
        self.majority_class       = maj
        # hack so all pydot vertices are unique
        self.count                = DTreeNode.i
        self.classification_error = 0
        self.num_seen             = 0
        self.prune_error          = 0
        self.pruned               = False
        DTreeNode.i += 1

    @staticmethod
    def byCategory(category):
        return DTreeData(category, -1, category)

    @property
    def leaf_description(self):
        """Return string description of leaf node"""
        return "%s\n%d   %d\n<%d>" % (self.value,
                                      self.classification_error,
                                      self.prune_error, self.count)
    @property
    def description(self):
        """Return string description of non-leaf node"""
        return "%s < %f\n%d   %d\n<%d>" % (self.col,
                                           self.value,
                                           self.classification_error,
                                           self.prune_error, self.count)
    def prune(self):
        """Modify data to reflect pruning"""
        self.value  = self.majority_class
        self.col    = -1

class DTree(BinaryTree):

    BY_GAIN = 1
    BY_FREQ = 2
    BY_BOTH = 3
    MIN_GAIN = 0.125
    MIN_FREQ = 0.9

    def __init__(self, data, left=None, right=None):
        super().__init__(data, left, right)

    @staticmethod
    def learn_decision_tree(dataset, method=DTree.BY_GAIN):
        """
        Given dataframe, learn decision tree using optimal information gain
        """
        if num_groups(dataset) == 1:
            category = re.split('[ \t\n\r]+', repr(dataset['genre']))[1]
            return DTree(DTreeData.byCategory(category))
        gain,axis,threshold = find_optimal_split(dataset)
        max_col, max_val = getMajorityClass(dataset)
        N = len(dataset)
        if method == DTree.BY_BOTH:
            if gain < DTree.MIN_GAIN and float(max_val) / N > DTree.MIN_FREQ:
                return DTreeData.byCategory(max_col)
        elif method == DTree.BY_GAIN:
            if gain < DTree.MIN_GAIN:
                return DTreeData.byCategory(max_col)
        elif method == DTree.BY_FREQ:
            if max_val * 1.0 / N > DTree.MIN_FREQ:
                return DTreeData.byCategory(max_col)
        left_subset  = dataset[dataset[axis] <  threshold]
        left_node    = learn_decision_tree(left_subset)
        right_subset = dataset[dataset[axis] >= threshold]
        right_node   = learn_decision_tree(right_subset)
        return DTreeNode(DTreeData(threshold, axis, max_col),
                         left_node,
                         right_node)

    def writeGraph(self):
        """Returns pydot graphical representation of tree"""
        def toGraph_(node, graph):
            if node.pruned:
                color = "cyan"
            elif node.isLeaf:
                color = genre_map[node.value]
            else:
                color = feature_map[node.col]
            if node.isLeaf:
                desc = node.leaf_description
            else:
                desc = node.description
            vertex = pydot.Node(desc, style="filled", fillcolor=color)
            graph.add_node(vertex)
            if not node.isLeaf:
                lNode = toGraph_(node.left,  graph)
                rNode = toGraph_(node,right, graph)
                graph.add_node(lNode)
                graph.add_node(rNode)
                graph.add_edge(pydot.Edge(vertex, lNode))
                graph.add_edge(pydot.Edge(vertex, rNode))
            return vertex
        graph = pydot.Dot(graph_type='digraph', ordering='out')
        toGraph_(self, graph)
        graph.write_png(file_)

    def decide (self, datarow):
        """Decide genre for given song"""
        truth = datarow['genre']
        self.num_seen += 1
        if self.isLeaf:
            if DEBUG:
                print('Categorized as', self.value)
            ret = self.value
        elif datarow[self.col] < self.value:
            if DEBUG:
                print('Going left  at %s < %f' % (datarow[self.col], self.value))
            ret = self.left.decide(datarow)
        else:
            if DEBUG:
                print('Going right at %s < %f' % (datarow[self.col], self.value))
            ret = self.right.decide(datarow)
        if ret != truth:
            if DEBUG:
                print('Incorrect: expected %s, received %s' % (truth, ret))
                print('Incrementing class error (%s)' % truth)
            self.classification_error += 1
        if self.majority_class != truth:
            if DEBUG:
                print('Incrementing prune error (%s)' % truth)
            self.prune_error += 1
        return ret

    def getMaxReducingNode(self):
        """
        For post-prune purposes, traverse annotated tree to find the node
        that when pruned to a leaf, most reduces the classification error
        """
        if self.isLeaf:
            return self.data
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
        """
        Prunes Tree to have no children
        Value takes on most abundant genre
        """
        self.node.prune()
        self.left   = None
        self.right  = None
        self.pruned = True

    def post_prune(self, df):
        """Post-prune tree until no further reduction in classification error"""
        diff = 0
        while True:
            for _, row in df.iterrows():
                if DEBUG:
                    print('Using', row['ID'], row['genre'])
                self.decide(row)
            node = self.getMaxReducingNode()
            if DEBUG:
                print(node.classification_error - node.prune_error)
            if node.classification_error - node.prune_error > diff:
                node.prune()
                diff = node.classification_error - node.prune_error
            else:
                break

    def prettyPrint(self, indent=0):
        """Pretty prints tree into formatted if-else python code"""
        s = ' ' * indent
        width = 2
        if self.isLeaf:
            print('%sreturn %s' % (s, node.value)
        else:
            print('%sif %s < %s:' % (s, node.col, node.value))
            self.left.prettyPrint(indent + width)
            print('%selse: # %s %s' % (s, node.col, node.value))
            self.right.prettyPrint(indent + width)
