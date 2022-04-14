import numpy as np
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd

features = pd.read_csv("train_features.csv").to_numpy()
labels = pd.read_csv("train_labels.csv").to_numpy()

features = features[:,2:]
labels = labels[:,1:]

#split data into folds keep last tenth as validation fold
fold_size = int(len(labels)/10)
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


feature_valid_folds = feature_folds[9]
label_valid_folds = label_folds[9]
# print(f"size of valid features: {feature_valid_folds.shape}")
# print(f"size of valid labels: {label_valid_folds.shape}")
# print(f"valid labels: {label_valid_folds}")


clf = svm.SVC(kernel='linear')
clf.fit(x_train,y_train)



