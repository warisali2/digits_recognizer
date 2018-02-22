from read_data import read, show

features_labels, features_train = read(dataset="training")
test_labels, test_train = read(dataset="testing")