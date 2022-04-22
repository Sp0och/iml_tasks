from sys import call_tracing
import numpy as np
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GridSearchCV
from itertools import zip_longest
from torch import row_stack

features = pd.read_csv("input_data/train_features.csv")
labels = pd.read_csv("input_data/train_labels.csv")

## TODO split data in chunks of 12
'''
pids = features["pid"].unique()
for pid in pids:
    # for each test in data
    for test in features.columns[1:]:
        # get 12 datapoints for this test as a row
        test_data = features[features["pid"] == pid][test].to_numpy()

print(test_data)
'''

## TODO Take one Test of one patient and transpose. concatenate with all other  to the matrix pid x1 x2 ...x12

## TODO calculate mean of one patient in one row

## TODO Replace all nan wit the mean of one row
#for row in rows
#    if np.isnan(row):
#        row = np.mean(row)

## TODO call svm.SVR for the fit

# pid x1 x2 x3..x12 
# y_LABEL

'''
vital_signs_ls = ['LABEL_RRate', 'LABEL_ABPm' , 'LABEL_SpO2', 'LABEL_Heartrate']
regressors_vital_signs = []
X = train_features_preprocessed
parameters = [5,10,5,20]
for idx,label in enumerate(vital_signs_ls):
  y = train_labels[label]
  c = parameters[idx]
  regressor = svm.SVR(C = c)
  fitted_regressor = regressor.fit(X,y)
  regressors_vital_signs.append(fitted_regressor)
'''