import numpy as np
from sklearn import svm
import pandas as pd
from alive_progress import alive_bar
from sklearn.model_selection import train_test_split


TEST_LABELS = ['LABEL_RRate', 'LABEL_ABPm' , 'LABEL_SpO2', 'LABEL_Heartrate']
VITAL_SIGNS = ['RRate', 'ABPm' , 'SpO2', 'Heartrate']

# reg_dict ist der predictor
def predict_on_test_data(reg_dict, feature_df):
    # gehe durch df (DataFrame) und wandle dieses in matrix um
    predicitions = feature_df["pid"].unique()
    for vital in VITAL_SIGNS:
        features = feature_df[vital].to_numpy()
        features = features.reshape((-1,12))
        reg = reg_dict[vital]
        pred = reg.predict(features)
        predicitions = np.column_stack((predicitions, pred))
    pred_df = pd.DataFrame(predicitions, columns=['pid'] + TEST_LABELS)
    return pred_df

# predict on all features - performed worse
# def predict_on_all_test_data(reg_dict, feature_df):
#     features = feature_df.to_numpy()
#     # remove pid from features
#     features = features[:, 1:]
#     # gehe durch df (DataFrame) und wandle dieses in matrix um
#     predicitions = feature_df["pid"].unique()
#     for vital in VITAL_SIGNS:
#         reg = reg_dict[vital]
#         pred = reg.predict(features)
#         predicitions = np.column_stack((predicitions, pred))
#     pred_df = pd.DataFrame(predicitions, columns=['pid'] + TEST_LABELS)
#     return pred_df

def train_model(feature_df, label_df):
    reg_dict = {}
    regressor_c = [5,10,5,20]
    with alive_bar(len(VITAL_SIGNS)) as bar:
        for idx,vital in enumerate(VITAL_SIGNS):
            features = feature_df[vital].to_numpy()
            features = features.reshape((-1,12))
            labels = label_df[TEST_LABELS[idx]].to_numpy()
            x_train,x_valid,y_train,y_valid = train_test_split(features,labels, train_size=0.9, random_state=42)
            # reg = svm.LinearSVR(C=regressor_c[idx],max_iter=10000, loss='squared_epsilon_insensitive', dual=False)
            np.random.seed(42)
            reg = svm.SVR(C = regressor_c[idx], kernel="rbf")
            reg.fit(x_train,y_train)
            reg_dict[vital] = reg
            print("Score for " + str(vital) + ": " + str(reg.score(x_valid,y_valid)))
            bar()
    return reg_dict

# train in all features and trends instead of all data from vital - performed wors
# def train_model_on_all(feature_df, label_df):
#     feature_df.sort_values(by=["pid"], inplace=True)
#     features = feature_df.to_numpy()
#     # remove pid from features
#     features = features[:, 1:]

#     label_df.sort_values(by=["pid"], inplace=True)

#     reg_dict = {}
#     regressor_c = [5,10,5,20]
#     with alive_bar(len(VITAL_SIGNS)) as bar:
#         for idx,vital in enumerate(VITAL_SIGNS):
#             labels = label_df[TEST_LABELS[idx]].to_numpy()
#             x_train,x_valid,y_train,y_valid = train_test_split(features,labels, train_size=0.9)
#             reg = svm.SVR(C = regressor_c[idx])
#             reg.fit(x_train,y_train)
#             reg_dict[vital] = reg
#             print("Score for " + str(vital) + " " + str(reg.score(x_valid,y_valid)))
#             bar()
#     return reg_dict