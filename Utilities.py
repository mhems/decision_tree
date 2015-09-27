import math

TARGET = 'genre'

def entropy(array):
    """
    Given array-like structure, compute its entropy
    """
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
    y = list(dataset[TARGET])
    ye = entropy(y)
    a = list(dataset[dataset[axis] <  threshold][TARGET])
    ae = (1.0 * len(a)/len(y)) * entropy(a)
    b = list(dataset[dataset[axis] >= threshold][TARGET])
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
    """
    Return number of genres in dataset
    """
    return len(dataset.groupby(TARGET).groups)

def getMajorityClass(dataset):
    """
    Given dataset, return most abundant genre
    """
    grps = dataset.groupby(TARGET).groups
    max_val = 0
    for cat in grps:
        if len(grps[cat]) > max_val:
            max_val = len(grps[cat])
            max_col = cat
    if DEBUG:
        print('Majority class %s (%d)' % (max_col, max_val))
    return max_col, max_val
