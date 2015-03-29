#!/usr/bin/python2

##################
# Driver methods
##################

def rec_print(node, indent):
    s = ' ' * indent
    if (node.left != None):
        print (s + 'if %s < %f:' % (node.col, node.value))
        rec_print(node.left, indent + 2)
        print (s + 'else:' % (node.col, node.value))
        rec_print(node.right, indent + 2)
    else:
        print s + 'return ' + repr(node.value)

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
    print '%d incorrect out of %d (%.2f%%)' % (wrong,total,wrong * 100 / total)

if __name__ == '__main__':
    #rec_print(dTree.root, 0)
    test_tree(SALAMI_path + 'our_data/train_first.csv',
              SALAMI_path + 'our_data/test_first.csv' )

