import numpy as np
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
# import os
from sklearn.model_selection import GridSearchCV
# os.chdir("task2")

features = pd.read_csv("output_data/train_features_processed.csv").to_numpy()
labels = pd.read_csv("input_data/train_labels.csv")
labels = labels.sort_values(by="pid").to_numpy()


features = features[:,2:]
labels = labels[:,11]

#split data into folds keep last tenth as validation fold
fold_size = int(len(labels)/10)
print(f"fold size: {fold_size}, label size: {len(labels)}")
split_idx = np.linspace(fold_size, 9*fold_size, 9).astype(int)
#10 folds of equal size
feature_folds = np.split(features,split_idx,axis=0)
label_folds = np.split(labels,split_idx,axis=0)

feature_train_folds = feature_folds[0:9]
label_train_folds = label_folds[0:9]
x_train = np.concatenate(feature_train_folds,axis=0)
y_train = np.concatenate(label_train_folds,axis=0)
print(f"shape of train X = {x_train.shape}")
print(f"shape of train Y = {y_train.shape}")


x_valid = feature_folds[9]
y_valid = label_folds[9]
# print(f"size of valid features: {feature_valid_folds.shape}")
# print(f"size of valid labels: {label_valid_folds.shape}")
# print(f"valid labels: {label_valid_folds}")

# clf = svm.SVC(kernel='linear')
# clf.fit(x_train,y_train)
# y_pred = clf.predict(x_valid)
# print(f"Accuracy SVC with valid being last: {metrics.accuracy_score(y_valid,y_pred)}")

#get the test label predictions
# test_features = pd.read_csv("test_features.csv").to_numpy()
# test_features = test_features[:,2:]
# y_test = clf.predict(test_features)
# np.savetxt("Sepsis_predictions.csv",y_test,delimeter=',')


clf = svm.LinearSVC(dual=False, fit_intercept=False, class_weight='balanced', loss="squared_epsilon_insensitive")
clf.fit(x_train,y_train)
# param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1.]}
# grds = GridSearchCV(clf, param_grid, n_jobs=4)
# grds.fit(x_train, y_train)
# y_pred = grds.predict(x_valid)
y_pred = clf.predict(x_valid)
print(f"linear SVC Accuracy with valid being last: {metrics.accuracy_score(y_valid,y_pred)}")
n_errors = np.sum(abs(y_pred-y_valid))
print(f"manual error rate with valid being last: {n_errors/len(y_valid)}")





