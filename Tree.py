import pandas as pd
import pydot

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
        """
        Factory method to construct a TreeNode with no children
        """
        return cls(val, -1, val, None, None)

    def decide (self, datarow):
        """
        Given a datarow of a song, run data through node subtree to decide the song's genre
        """
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
        """
        Prunes node to have no children
        Value takes on most abundant genre 
        """
        self.value  = self.majority_class
        self.col    = -1
        self.left   = None
        self.right  = None
        self.pruned = True

    @property
    def isLeaf(self):
        """
        Returns True iff node has no children
        """
        return self.left is None and self.right is None

    @property
    def description(self):
        """
        Returns string description of node
        """
        DTreeNode.i += 1
        if self.isLeaf():
            return "%s\n%d   %d\n<%d>" % (self.value,
                                          self.classification_error,
                                          self.prune_error, self.count)
        else:
            return "%s < %f\n%d   %d\n<%d>" % (self.col,
                                               self.value,
                                               self.classification_error,
                                               self.prune_error, self.count)

    @property
    def height(self):
        """
        Returns height of self's subtree
        """
        if self.isLeaf:
            return 0
        return 1 + max(self.left.height, self.right.height)

    @property
    def size(self):
        """
        Returns number of vertices in self's subtree
        """
        if self.isLeaf:
            return 1
        return 1 + self.left.size + self.right.size

    def toGraph(self, graph):
        """
        Converts self subtree to pydot graph
        """
        myNode = self.getVertex()
        graph.add_node(myNode)
        if self.isLeaf:
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
        """
        Returns pydot vertex representation of self
        """
        if self.pruned:
            color = "cyan"
        elif self.isLeaf:
            color = genre_map[self.value]
        else:
            color = feature_map[self.col]            
        return pydot.Node(self.getDescription(),
                          style="filled",
                          fillcolor=color)

    def prettyPrint(self, indent):
        """
        Pretty prints self subtree into if-else python code
        """
        s = ' ' * indent
        if self.isLeaf:
            print(s + 'return ' + repr(self.value))
        else:
            print(s + 'if %s < %f:' % (self.col, self.value))
            self.left.prettyPrint(indent + 2)
            print(s + 'else: # %s %f' % (self.col, self.value))
            self.right.prettyPrint(indent + 2)

class DTree:

    def __init__(self, filename, val_fn=None):
        dataframe = pd.read_csv(filename)
        dataframe = getRelevantFeatures(dataframe)
        self.root = learn_decision_tree(dataframe)
        if DEBUG:
            print('Decision tree for %s has been learned' % getBaseName(filename))
            print('Height: %d Vertices: %d' % (self.height, self.size))
        if val_fn is not None:
            self.post_prune(pd.read_csv(val_fn))
            if DEBUG:
                print('Pruned height: %d Pruned size: %d' % (self.height,
                                                             self.size))
        if SAVE_GRAPH:
            self.toGraph().write_png('%s.png' % filename)

    def toGraph(self):
        """
        Returns pydot graphical representation of tree
        """
        graph = pydot.Dot(graph_type='digraph', ordering='out')
        self.root.toGraph(graph)
        return graph

    @property
    def height(self):
        """
        Returns height of tree
        """
        return self.root.height

    @property
    def size(self):
        """
        Returns number of vertices in tree
        """
        return self.root.size

    def decide (self, datarow):
        """
        Decide genre for given song
        """
        return self.root.decide(datarow)

    def post_prune(self, df):
        """
        Post-prune tree until no further reduction in classification error
        """
        diff = 0
        while True:
            for _, row in df.iterrows():
                if DEBUG:
                    print('Using', row['ID'], row['genre'])
                self.decide(row)
            node = self.root.getMaxReducingNode()
            if DEBUG:
                print(node.classification_error - node.prune_error)
            if node.classification_error - node.prune_error > diff:
                node.pruned = True
                node.prune()
                diff = node.classification_error - node.prune_error
            else:
                break

    def prettyPrint(self):
        """
        Pretty prints tree into if-else python code
        """
        self.root.prettyPrint(0)
