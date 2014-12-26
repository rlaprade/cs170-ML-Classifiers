

data_dir = "./../hw12data/emailDataset"
val_labels = data_dir + "/valLabels.csv"

for T in (1,2, 5, 10, 25):
    with open("./../emailOutput{}.csv".format(T), 'r') as out:
        with open(val_labels, 'r') as v_labels:
            val = v_labels.readlines()
            ours = out.readlines()
            errors = 0
            for i in range(len(val)):
                if val[i] != ours[i]:
                    errors += 1
        print("T={T} Error rate: {e}%".format(T=T, e=float(errors)*100/len(val)))