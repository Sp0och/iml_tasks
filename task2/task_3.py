import numpy as np
from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV
import pandas as pd
from alive_progress import alive_bar
from sklearn.model_selection import train_test_split


TEST_LABELS = ['LABEL_RRate', 'LABEL_ABPm' , 'LABEL_SpO2', 'LABEL_Heartrate']
vital_signs = ['RRate', 'ABPm' , 'SpO2', 'Heartrate']

# reg_dict ist der predictor
def predict_on_test_data(reg_dict, feature_df):
    # gehe durch df (DataFrame) und wandle dieses in matrix um
    predicitions = feature_df["pid"].unique()
    for vitals in vital_signs:
        features = feature_df[vitals].to_numpy()
        features = features.reshape((-1,12))
        reg = reg_dict[vitals]
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
    parameters = [5,10,5,20]
    for idx,vitals in enumerate(vital_signs):
        features = feature_df[vitals].to_numpy()
        features = features.reshape((-1,12))
        labels = label_df[TEST_LABELS[idx]].to_numpy()
        # train svm without temporal information
        x_train,x_valid,y_train,y_valid = train_test_split(features,labels, train_size=0.9)
        c = parameters[idx]
        reg = svm.LinearSVR(C=c,max_iter=10000, loss='squared_epsilon_insensitive', dual=False)
        reg.fit(x_train,y_train)
        reg_dict[vitals] = reg
        # y wollen wir damit nun predicten, mit y_valid checken wir das nachher
        y_predict = reg.predict(x_valid)
        error = np.sqrt(np.mean((y_predict-y_valid)**2))
        print('Error for ' + vitals + ": "  + str(error))
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