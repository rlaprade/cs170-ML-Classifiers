import digit_classifier as dc
import parser

data_dir = "./../hw12data/digitsDataset"
test_data = data_dir + "/testFeatures.csv"
training_data = data_dir + "/trainFeatures.csv"
training_labels = data_dir + "/trainLabels.csv"
   
t_feats = []
t_labels = []
with open(training_data,'r') as training_dataset:
    with open(training_labels,'r') as training_labelset:
        trainingset = training_dataset.readlines()
        t_labels = parser.parse_labels(training_labelset.readlines())
        for i in range(len(trainingset)):
            t_feats.append(parser.parse_features(trainingset[i]))

k=1
write_out = ""
with open(test_data, 'r') as dataset:
    for digit in dataset.readlines():
        label = dc.classify_digit(parser.parse_features(digit), k,t_feats,t_labels)
        write_out += "{}\n".format(label)
with open("./../digitsOutput.csv".format(k), 'w') as out:
   out.write(write_out[:-1])
