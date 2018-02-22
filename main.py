from read_data import read, show

features_labels, features_train = read(dataset="training")
labels_test, features_test = read(dataset="testing")

from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()

clf.fit(features_train, features_labels)

labels_pred = clf.predict(features_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(labels_test, labels_pred))
