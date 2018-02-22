from read_data import read, show

# Reading training and testing features from dataset
features_labels, features_train = read(dataset="training")
labels_test, features_test = read(dataset="testing")

# Reshaping features_train and features_test as sklearn
# requires 2D array in fit()
nsamples, nx, ny = features_train.shape
features_train = features_train.reshape((nsamples,nx*ny))

nsamples, nx, ny = features_test.shape
features_test = features_test.reshape((nsamples,nx*ny))

# NaiveBayes Model
from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()

clf.fit(features_train, features_labels)

labels_pred = clf.predict(features_test)

from sklearn.metrics import accuracy_score

print("NB: ", accuracy_score(labels_test, labels_pred))

# SVM
from sklearn.svm import SVC

clf = SVC(kernel='rbf')

clf.fit(features_train, features_labels)

labels_pred = clf.predict(features_test)
