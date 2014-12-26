import parser
import random
from math import log


NUM_FEATURES = 57
LABELS = (0, 1)
   
class DecisionTreeNode(object):
    def __init__(self, feature, threshhold, left, right, label=None):
        self.feature = feature
        self.left = left
        self.right = right
        self.label = label
        self.threshhold = threshhold
        assert (type(left) == DecisionTreeNode and type(right) == DecisionTreeNode) or label != None
        
    def classify(self, obs):
        """ Classifies given observation obs (a list of features).
        Classifies recursively if this node is internal.
        """
        if self.label != None:
            return self.label
        f = obs[self.feature]
        if f <= self.threshhold:
            return self.left.classify(obs)
        else:
            return self.right.classify(obs)
      
      
class DataPoint(object):
    def __init__(self, feat_list, label):
        self.features = feat_list
        self.label = label
        
    def __str__(self):
        return "{l}: {f}".format(f=self.features, l=self.label)
        
    def __repr__(self):
        return "{l}: {f}".format(f=self.features, l=self.label)
            
            
def split(data, feat, threshhold):
    """ Splits the data (a list of lists of features)
    into two sets based on the threshhold for feature at index feat
    """
    l, r = [], []
    for obs in data:
        if obs.features[feat] <= threshhold:
            l.append(obs)
        else:
            r.append(obs)
    return l, r
    
def goodness(split):
    """ Measures the goodness of a binary split of a
    dataset
    """
    l, r = split
    s = l + r
    return imp(s) - (len(l)/len(s) * imp(l) + len(r)/len(s) * imp(r))
    
def imp(set):
    """ Calculates the impurity (entropy) of the given set"""
    P = {}
    for l in LABELS:
        P[l] = 0
    for datum in set:
        P[datum.label] += 1.0/len(set)
    return -sum(P[l]*log(P[l]) for l in LABELS if P[l] > 0)
    
def stop(data, depth):
    """ Returns boolean of whether or not the data set
    meets our stop conditions
    
    Stop Condidions:
    1.  100% homogenous
    2.  depth >= 10
    3.  entropy(data) < 0.3
    """
    labels = set()
    for point in data:
        labels.add(point.label)
    if len(labels) <= 1 or depth >= 10 or imp(data) < 0.3:
        return True
    return False
       
def bag(training_set):
    """ Selelct a subset of the training set (with possible repetition)
    to use for  a decision tree. Subset size will be .632 of set size
    """
    subset_size = int(.632 * len(training_set))
    subset = []
    for _ in range(subset_size):
        subset.append(random.choice(training_set))
    return subset
    
def feature_sample(num_features):
    """ Select a subset of features. These will be candidates
    for a particular node.
    """
    subset_size = 8
    select_from = list(range(num_features))
    subset = []
    for _ in range(subset_size):
        i = random.randint(0,len(select_from)-1)
        f = select_from.pop(i)
        subset.append(f)
    return subset
    
def choose_label(dataset):
    """ Choose a label for the given dataset"""
    label_count = {}
    for l in LABELS:
        label_count[l] = 0
    for datum in dataset:
        label_count[datum.label] += 1
    return max(label_count, key=lambda key: label_count[key])
    
def extract_train_data(training_data, training_labels):
    """ Returns a list of datapoints for the 
    given data and labels.
    """
    train_set = []
    train_labels = []
    with open(training_data,'r') as training_dataset:
        with open(training_labels,'r') as training_labelset:
            trainingset = training_dataset.readlines()
            train_labels = parser.parse_labels(training_labelset.readlines())
            for i in range(len(trainingset)):
                obs = DataPoint(parser.parse_features(trainingset[i]), train_labels[i])
                train_set.append(obs)
    return train_set
    
def threshhold_cands(train_set):
    """ Returns a list where each index i of threshholds
    is a list of all candidate threshholds for feature i
    """
    threshholds = []
    for i in range(NUM_FEATURES):
        fs = [train_set[j].features[i] for j in range(len(train_set))]
        fs.sort()
        candidates = [(fs[j] + fs[j+1])/2 for j in range(len(fs)-1)]
        threshholds.append(candidates)
    return threshholds
            
def build_tree(training_set, thresh_cands):
    """ Returns a random decision tree for the given
    training set """
    def build_subtree(train_set, depth=0):
        if stop(train_set, depth):
            return DecisionTreeNode(None, None, None, None, choose_label(train_set))
        feature_set = feature_sample(NUM_FEATURES)
        decision_points = ((feature_set[i], t) for i in range(len(feature_set)) for t in thresh_cands[i])
        chosen_feature, chosen_threshold = max(decision_points, key=lambda x: goodness(split(train_set, x[0], x[1])))
        l, r = split(train_set, chosen_feature, chosen_threshold)
        left = build_subtree(l, depth+1)
        right = build_subtree(r, depth=1)
        node = DecisionTreeNode(chosen_feature, chosen_threshold, left, right)
        return node        
    dataset = bag(training_set)
    return build_subtree(dataset)

def build_forest(training_set, thresh_cands, T):
    """ Builds a random forest of T decision trees """
    forest = []
    for _ in range(T):
        forest.append(build_tree(training_set, thresh_cands))
    return forest
    
def forest_classify(forest, obs):
    """ Returns a label for the given observation obs
    as assigned by the given random forest. (Majority 
    vote of constituent trees)
    Ties will be decided randomly.
    """
    votes = {}
    for l in LABELS:
        votes[l] = 0
    for tree in forest:
        l = tree.classify(obs)
        votes[l] += 1
    return max(votes.keys(), key=(lambda key: votes[key]))
    # leader = None
    # for l in LABELS:
        # if leader == None:
            # leader = [l]
        # else:
            # if votes[l]>leader:
                # leader = [l]
            # elif votes[l]==leader:
                # leader.append(l)
    # return random.choice(leader)

# for k in (1,2, 5, 10, 25):
    # labels = []
    # write_out = ""
    # with open(val_data, 'r') as dataset:
        # for digit in dataset.readlines():
            # label = dc.classify_digit(parser.parse_features(digit), k,train_feats,train_labels)
            # labels.append(label)
            # write_out += "{}\n".format(label)
    # with open("./../digitsOutput{}.csv".format(k), 'w') as out:
       # out.write(write_out)
            
    # errors = 0
    # with open(val_labels, 'r') as true_labels:
        # tr_labels = parser.parse_labels(true_labels.readlines())
        # for i in range(len(labels)):
            # if tr_labels[i] != labels[i]:
                # errors += 1
    # print("k={k} Error rate: {e}%".format(k=k, e=float(errors)*100/len(labels)))
