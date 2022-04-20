import numpy as np
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
from alive_progress import alive_bar
from sklearn.model_selection import GridSearchCV


feature_df = pd.read_csv("normalized_train_features.csv", )
label_df = pd.read_csv("input_data/train_labels.csv")
feature_df.sort_values(by='pid')
label_df.sort_values(by=['pid'])
feature_df.set_index('pid')
# merge them on pid
# labeled_features.set_index('pid').join(labels.set_index('pid'))

test_label_names = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 
    'LABEL_Alkalinephos', 'LABEL_Bilirubin_total', 'LABEL_Lactate', 
    'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct', 
    'LABEL_EtCO2']

# train svm without temporal information
with alive_bar(len(test_label_names)) as bar:
  for test in test_label_names:
    labels = label_df[test].to_numpy()
    features = feature_df.to_numpy()
    features = features[:, 1:]

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
    print(f"shape of valid X = {feature_valid_folds.shape}")
    print(f"shape of valid Y = {label_valid_folds.shape}")
    # clf = svm.SVC(kernel='linear')
    # TODO figure out which one is nice 
    clf = svm.SVC(kernel='linear', C=0.1)
    clf.fit(x_train, y_train)
    # clf = svm.LinearSVC(dual=False, fit_intercept=False, verbose=0, class_weight='balanced')
    # param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1.]}
    # grds = GridSearchCV(clf, param_grid, n_jobs=4, verbose=0)
    # grds.fit(x_train, y_train)

    predicted_labels = clf.predict(feature_valid_folds)
    n_errors = np.sum(abs(predicted_labels-label_valid_folds))
    print(n_errors/len(label_valid_folds))
    bar()



