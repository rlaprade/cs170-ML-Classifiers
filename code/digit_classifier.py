import sys
import heapq
from math import sqrt

data_dir = "./../hw12data/digitsDataset"
training_data = data_dir + "/trainFeatures.csv"
t_labels = data_dir + "/trainLabels.csv"

def classify_digit(digit, k, training_data, training_labels):
    """ Classifies digit specified by features
    given as read in line "digit" using 
    k-nearest neighbors algorithm
    """
    nearest = []  #heap of k nearest neighbors
    for i in range(len(training_data)):
        dist = feature_dist(training_data[i], digit)
        if len(nearest) < k:
            heapq.heappush(nearest, (-dist, training_labels[i]))
        else:
            heapq.heappushpop(nearest, (-dist, training_labels[i]))
    votes = {}
    nearest.sort()
    for neighbor in nearest:
        label = neighbor[1]
        if label in votes:
            votes[label] += 1
        else:
            votes[label] = 1
    return max(votes.keys(), key=(lambda key: votes[key]))
            
def sq_diff(x, y):
    a = x - y
    return a*a
    
def feature_dist(x, y):
    """ Computes the distance between features
    x and features y
    """
    sq_sum = 0
    for i in range(len(x)):
        sq_sum += sq_diff(x[i], y[i])
    return sqrt(sq_sum)
   
# print(classify_digit(sys.argv[1], sys.argv[2]))

