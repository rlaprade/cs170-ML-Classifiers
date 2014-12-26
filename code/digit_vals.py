import digit_classifier as dc
import parser

data_dir = "./../hw12data/digitsDataset"
val_data = data_dir + "/valFeatures.csv"
val_labels = data_dir + "/valLabels.csv"
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

for k in (1,2, 5, 10, 25):
    labels = []
    write_out = ""
    with open(val_data, 'r') as dataset:
        for digit in dataset.readlines():
            label = dc.classify_digit(parser.parse_features(digit), k, t_feats, t_labels)
            labels.append(label)
            write_out += "{}\n".format(label)
    with open("./../digitsOutput{}.csv".format(k), 'w') as out:
       out.write(write_out[:-1])
            
    errors = 0
    with open(val_labels, 'r') as true_labels:
        tr_labels = parser.parse_labels(true_labels.readlines())
        for i in range(len(labels)):
            if tr_labels[i] != labels[i]:
                errors += 1
                # print("Misclassified {i}th digit as {fake} instead of {real}".format(i=i, fake=labels[i], real=tr_labels[i]))
    print("k={k} Error rate: {e}%".format(k=k, e=float(errors)*100/len(labels)))
