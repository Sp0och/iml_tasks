import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import pandas as pd
from alive_progress import alive_bar
from sklearn.linear_model import LogisticRegression

TEST_LABELS = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 
      'LABEL_Alkalinephos', 'LABEL_Bilirubin_total', 'LABEL_Lactate', 
      'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct', 
      'LABEL_EtCO2', 'LABEL_Sepsis']

def predict_on_test_data(clf_dict, feature_df):
  features = feature_df.to_numpy()
  # init with pids
  predicted_probabilities = features[:, 0]
  # remove pid from features
  features = features[:, 1:]
  for test in TEST_LABELS:  
    clf = clf_dict[test]
    pred =  clf.predict_proba(features)[:, 1]
    predicted_probabilities = np.column_stack((predicted_probabilities, pred))
  # convert back to df
  prob_df = pd.DataFrame(predicted_probabilities, columns=['pid'] + TEST_LABELS)
  return prob_df

def train_model(feature_df, label_df):
  feature_df.set_index('pid')
  clf_dict = {}
  # train svm without temporal information
  with alive_bar(len(TEST_LABELS)) as bar:
    for test in TEST_LABELS:
      labels = label_df[test].to_numpy()
      features = feature_df.to_numpy()
      # UNCOMMENT FOR TESTING WITH LESS FEATURES:
      # labels = label_df[test].to_numpy()[0:100]
      # features = feature_df.to_numpy()[0:100,:]

      # remove pid from features
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

      # print(f"shape of train X = {x_train.shape}")
      # print(f"shape of train Y = {y_train.shape}")

      feature_valid_folds = feature_folds[9]
      label_valid_folds = label_folds[9]
      # print(f"shape of valid X = {feature_valid_folds.shape}")
      # print(f"shape of valid Y = {label_valid_folds.shape}")

      # TODO figure out which one is nice 
      # clf = svm.SVC(kernel='linear', C=0.1)
      # clf.fit(x_train, y_train)

      svm_ = svm.LinearSVC(dual=False, fit_intercept=False, verbose=0, class_weight='balanced')
      # param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1.]}
      # grd = GridSearchCV(svm_, param_grid, n_jobs=4, verbose=0)
      # grd.fit(x_train, y_train)
      # print("Best Parameters: ")
      # print(grd.best_params_)
      # clf = CalibratedClassifierCV(grd.best_estimator_)
      clf = CalibratedClassifierCV(svm_)
      clf.fit(x_train, y_train)
      clf_dict[test] = clf
      predicted_probabilities = clf.predict_proba(feature_valid_folds)
      # label = 1 if probability of 1 > 0.5 else 0
      predicted_labels = predicted_probabilities[:,1] > 0.5
      n_errors = np.sum(abs(predicted_labels-label_valid_folds))
      print("Accuracy for " + str(test))
      print(1.0 - n_errors/len(label_valid_folds))
      bar()
  return clf_dict
  



