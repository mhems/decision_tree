import math
import random

def entropy(array):
    """Given array-like structure, compute its entropy"""
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
    """
    Given dataframe, column axis, and split threshold,
    compute information gain of splitting axis column of dataframe with threshold
    """
    y = list(dataset['genre'])
    ye = entropy(y)
    a = list(dataset[dataset[axis] <  threshold]['genre'])
    ae = (1.0 * len(a)/len(y)) * entropy(a)
    b = list(dataset[dataset[axis] >= threshold]['genre'])
    be = (1.0 * len(b)/len(y)) * entropy(b)
    return ye - ae - be

def find_optimal_split(dataset):
    """
    Given dataframe, determine optimal split (axis and threshold)
    Uses a naive search to test each possible split on all axes
    Returns the resulting optimal gain, optimal axis, and optimal threshold
    """
    (best_gain,best_axis,best_threshold) = (0,0,0)
    axis_index = 0
    features = dataset.columns.tolist()[:-1]
    for axis in features:
        if axis is None or axis == '':
            continue
        uniq_data = dataset.sort(columns=axis,inplace=False)
        uniq_data = uniq_data.drop_duplicates(subset=axis)
        for index in range(0,len(uniq_data) - 1):
            datum1 = uniq_data.iloc[index,axis_index]
            datum2 = uniq_data.iloc[index + 1, axis_index]
            threshold = datum1 + (abs(datum1 - datum2)/2)
            gain = information_gain(dataset,axis,threshold)
            if gain > best_gain:
                (best_gain,best_axis,best_threshold) = (gain,axis,threshold)
        axis_index += 1
    return best_gain, best_axis, best_threshold

def num_groups(dataset):
    """Return number of genres in dataset"""
    return len(dataset.groupby('genre').groups)

def getMajorityClass(dataset):
    """Given dataset, return most abundant genre"""
    grps = dataset.groupby('genre').groups
    max_val = 0
    for cat in grps:
        if len(grps[cat]) > max_val:
            max_val = len(grps[cat])
            max_col = cat
    return max_col, max_val

def divyUpNumber(N, K):
    """Divy up, as evenly as possible, N things to K bins"""
    bins = [0 for _ in range(K)]
    while N > 0:
        for i in range(K):
            if N > 0:
                bins[i] += 1
            else:
                break
            N -= 1
    return bins

def getContiguousPartitions(lines, K):
    """Return K contiguous partitions of lines"""
    N = len(lines)
    bins = divyUpNumber(N, K)
    partitions = []
    start = 0
    for e in bins:
        partitions.append(lines[start:start+e])
        start += e
    return partitions

def getLinesRandomly(lines, amt):
    """Remove amt lines randomnly from lines and return it"""
    s = set()
    N = len(lines)
    while len(s) < amt:
        s.add(int(random.random() * N))
    indices = list(s)
    ret = [lines[i] for i in indices]
    indices.sort(reverse=True)
    for i in indices:
        lines.pop(i)
    return (ret, lines)

def getRandomPartitions(lines, K):
    """Get K random partitions of lines"""
    copy = lines
    partitions = []
    i = 0
    bins = divyUpNumber(N, K);
    for e in bins:
        t, copy = getLinesRandomly(copy, e)
        partitions.append(t)
    return partitions
