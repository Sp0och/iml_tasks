import numpy as np
from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import pandas as pd
from alive_progress import alive_bar
from sklearn.model_selection import train_test_split

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
  feature_df.sort_values(by=["pid"], inplace=True)
  label_df.sort_values(by=["pid"], inplace=True)
  clf_dict = {}
  # train svm without temporal information
  with alive_bar(len(TEST_LABELS)) as bar:
    for test in TEST_LABELS:
      # random sample equal number of data with label = 1 and label = 0
      # since there is less with label = 1, take all of them
      indexes = label_df.index
      label_1_indexes = indexes[label_df[test] == 1]
      label_0_indexes = indexes[label_df[test] == 0]
      np.random.seed(42)
      label_0_indexes = np.random.choice(label_0_indexes, size=len(label_1_indexes), replace=False)
      # concatenated indices for training
      train_indices = np.concatenate((label_1_indexes, label_0_indexes))
      np.random.shuffle(train_indices)
      labels = label_df[test].iloc[train_indices.tolist()].to_numpy()
      features = feature_df.iloc[train_indices.tolist()].to_numpy()
      # UNCOMMENT FOR TESTING WITH LESS FEATURES:
      # labels = label_df[test].to_numpy()[0:100]
      # features = feature_df.to_numpy()[0:100,:]

      # remove pid from features
      features = features[:, 1:]
      x_train, x_valid, y_train, y_valid = train_test_split(features,labels, train_size=0.9, random_state=42)
      
      svm_ = svm.LinearSVC(dual=False, fit_intercept=False, verbose=0, class_weight='balanced', max_iter=10000)
      if test == 'LABEL_SEPSIS':
        svm_.set_params(loss='squared_epsilon_insensitive')
      # param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1.]}
      # grd = GridSearchCV(svm_, param_grid, n_jobs=4, verbose=0)
      # grd.fit(x_train, y_train)
      # print("Best Parameters: ")
      # print(grd.best_params_)
      # clf = CalibratedClassifierCV(grd.best_estimator_)
      clf = CalibratedClassifierCV(svm_)
      clf.fit(x_train, y_train)
      clf_dict[test] = clf
      predicted_probabilities = clf.predict_proba(x_valid)
      # label = 1 if probability of 1 > 0.5 else 0
      predicted_labels = predicted_probabilities[:,1] > 0.5
      n_errors = np.sum(abs(predicted_labels-y_valid))
      print("Accuracy for " + str(test))
      print(1.0 - n_errors/len(y_valid))
      bar()
  return clf_dict
  



