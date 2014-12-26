import parser
import tree_decision as td

data_dir = "./../hw12data/emailDataset"
val_data = data_dir + "/valFeatures.csv"
val_labels = data_dir + "/valLabels.csv"
training_data = data_dir + "/trainFeatures.csv"
training_labels = data_dir + "/trainLabels.csv"


train_set = td.extract_train_data(training_data, training_labels)
threshholds = td.threshhold_cands(train_set)

for T in (1,2, 5, 10, 25):
    labels = []
    write_out = ""
    forest = td.build_forest(train_set, threshholds, T)
    with open(val_data, 'r') as dataset:
        for email in dataset.readlines():
            label = td.forest_classify(forest,parser.parse_features(email))
            labels.append(label)
            write_out += "{}\n".format(label)
    with open("./../emailOutput{}.csv".format(T), 'w') as out:
       out.write(write_out[:-1])
            
    errors = 0
    with open(val_labels, 'r') as v_labels:
        true_labels = parser.parse_labels(v_labels.readlines())
        for i in range(len(labels)):
            if true_labels[i] != labels[i]:
                errors += 1
    print("T={T} Error rate: {e}%".format(T=T, e=float(errors)*100/len(labels)))