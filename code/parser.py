def parse_features(line):
    """ Convert features from read in
    line to list of integers
    """
    if line[-1] == '\n': 
        line = line[:-1]
    return [float(f) for f in line.split(',')]
    
def parse_labels(l):
    """ Convert labels from read in
    line l to list of integers
    """
    if l[-1] == '\n': 
        l = l[:-1]
    return [int(label) for label in l]