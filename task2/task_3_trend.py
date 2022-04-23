import numpy as np
from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV
import pandas as pd
from alive_progress import alive_bar
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


TEST_LABELS = ['LABEL_RRate', 'LABEL_ABPm' , 'LABEL_SpO2', 'LABEL_Heartrate']

# reg_dict ist der predictor
def predict_on_test_data(reg_dict, feature_df):
    # gehe durch df (DataFrame) und wandle dieses in matrix um
    predicitions = feature_df["pid"]
    #exclude pid
    features = feature_df.to_numpy()[:,1:]
    for label in TEST_LABELS:
        #four different regs for the four labels
        reg = reg_dict[label]
        pred = reg.predict(features)
        predicitions = np.column_stack((predicitions, pred))
    pred_df = pd.DataFrame(predicitions, columns=['pid'] + TEST_LABELS)
    return pred_df
  
'''      
  # init with pids
  predicted_probabilities = features[:, 0]
  # remove pid from features
  features = features[:, 1:]
  
  
  # convert back to df
  column_names = TEST_LABELS
  column_names.insert(0, "pid")
  pred_df = pd.DataFrame(predicted_probabilities, columns=column_names)
    return pred_df
'''

def train_model(feature_df, label_df):
    reg_dict = {}
    # parameters = [5,10,5,20]
    features = feature_df.to_numpy()[:,1:]
    for idx,label in enumerate(TEST_LABELS):
        #consider columns of current vital
        #get all timesteps on one line
        labels = label_df[label].to_numpy()
        # train svm without temporal information
        x_train,x_valid,y_train,y_valid = train_test_split(features,labels, train_size=0.9)
        # c = parameters[idx]
        # reg = svm.LinearSVR(C=c,max_iter=10000, loss='squared_epsilon_insensitive', dual=False)
        reg = svm.LinearSVR(max_iter=10000, loss='squared_epsilon_insensitive', dual=False)
        param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1., 5, 10, 20]}
        grd = GridSearchCV(reg, param_grid, n_jobs=4, verbose=0)
        grd.fit(x_train, y_train)
        print("Best Parameters: ")
        print(grd.best_params_)
        # reg.fit(x_train,y_train)
        reg_dict[label] = grd
        # y wollen wir damit nun predicten, mit y_valid checken wir das nachher
        # y_predict = reg.predict(x_valid)
        y_predict = grd.predict(x_valid)
        error = np.sqrt(np.mean((y_predict-y_valid)**2))
        print('Error for ' + label + ": "  + str(error))
    return reg_dict
'''
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