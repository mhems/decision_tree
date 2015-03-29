##################
# Decision Tree Classes
##################

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

class DTree:
    def __init__(self, filename):
        dataframe = pd.read_csv(filename)
        dataframe = getRelevantFeatures(dataframe)
        self.root = learn_decision_tree(dataframe)
        print 'Decision tree for %s has been learned' % filename

    def decide (self, datarow):
        return self.decide_rec(self.root, datarow)

    def decide_rec(self, node, datarow):
        dec = node.decide(datarow)
        if type(dec) == type(""):
            return dec
        return self.decide_rec(dec, datarow)

