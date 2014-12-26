import parser
import tree_decision as td

data_dir = "./../hw12data/emailDataset"
test_data = data_dir + "/testFeatures.csv"
training_data = data_dir + "/trainFeatures.csv"
training_labels = data_dir + "/trainLabels.csv"


train_set = td.extract_train_data(training_data, training_labels)
threshholds = td.threshhold_cands(train_set)

T = 5

labels = []
write_out = ""
forest = td.build_forest(train_set, threshholds, T)
with open(test_data, 'r') as dataset:
    for email in dataset.readlines():
        label = td.forest_classify(forest,parser.parse_features(email))
        labels.append(label)
        write_out += "{}\n".format(label)
with open("./../emailOutput.csv".format(T), 'w') as out:
   out.write(write_out[:-1])
        