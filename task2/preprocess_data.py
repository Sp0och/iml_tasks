import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from alive_progress import alive_bar

# normal distributed data
NORM_COL_NAMES = ['SaO2', 'Fibrinogen', 'EtCO2', 'Temp', 'Hgb', 'HCO3', 
'BaseExcess', 'RRate', 'Phosphate', 'PaCO2', 'Platelets', 'Glucose', 'ABPm',
'Magnesium', 'Potassium', 'ABPd', 'Chloride', 'Hct', 'Heartrate', 'ABPs',
'pH', 'FiO2']

# exponentional distributed data
EXP_COL_NAMES = ['PTT', 'BUN', 'Lactate', 'WBC', 'Creatinine', 'AST', 
    'Alkalinephos', 'SpO2', 'Bilirubin_direct', 'Bilirubin_total',
    'TroponinI']

VITAL_SIGNS = ['RRate', 'ABPm' , 'SpO2', 'Heartrate']

def import_from_file(filename):
  data = pd.read_csv("./input_data/" + filename + ".csv")
  return data

class DataProcessor:
  def __init__(self):
    train_features_df = import_from_file("train_features")
    test_features_df = import_from_file("test_features")
    self.means = {}
    self.scales = {}
    print("[Starting] Finding scales and means")
    self.train_scaler(train_features_df)
    print("[Finished] Finding scales and means")
    print("[Starting] Appling feature transformations to train data")
    scaled_train_features = self.apply_scaler(train_features_df)
    print("[Finished] Appling feature transformations to train data")
    print("[Starting] Appling feature transformations to test data")
    scaled_test_features = self.apply_scaler(test_features_df)
    print("[Finished] Appling feature transformations to test data")
    print("[Starting] Processing train features for regression")
    reg_features = self.preproc_regression(train_features_df)
    reg_features.to_csv("output_data/train_reg_features.csv", index=False)
    print("[Finished] Processing train features for regression")
    print("[Starting] Processing test features for regression")
    reg_features = self.preproc_regression(test_features_df)
    reg_features.to_csv("output_data/test_reg_features.csv", index=False)
    print("[Finished] Processing test features for regression")
    print("[Starting] Preprocessing normalized train features")
    # test on partial dataset
    # processed_train_df = self.prepoc_df(scaled_train_features.iloc[0:1000, :])
    processed_train_df = self.prepoc_df(scaled_train_features)
    processed_train_df.to_csv("./output_data/train_features_processed.csv", index=True)
    print("[Finished] Preprocessing normalized train features")
    print("[Starting] Preprocessing normalized test features")
    processed_test_df = self.prepoc_df(scaled_test_features)
    processed_test_df.to_csv("./output_data/test_features_processed.csv", index=True)
    print("[Finished] Preprocessing normalized test features")

  # returns a dict of trained scalers
  def train_scaler(self, data):
    data_grouped = data.groupby(['pid'], as_index=False).mean()

    # select columns from df with column names col_names
    norm_features = data_grouped[NORM_COL_NAMES]

    # read colums with col_names from csv
    norm_data = norm_features.to_numpy().transpose()

    for col_data, col_name in zip(norm_data, norm_features.columns):
      # remove nan's from col_data
      data_corrected = col_data[~np.isnan(col_data)]
      if col_name == 'Fibrinogen' :
        data_corrected = np.log(data_corrected)
      if col_name == 'SaO2':
        data_corrected = 100 - data_corrected
        np.log10(data_corrected, out=data_corrected, where=data_corrected > 0)
      mean = np.mean(data_corrected)
      # remove outliers for calculating std_deviation
      data_filtered = data_corrected[abs(data_corrected - mean) < 2*np.std(data_corrected)]
      mean = np.mean(data_filtered)
      scale = np.std(data_filtered)
      self.means[col_name] = mean
      self.scales[col_name] = scale

    # handle exponential distributions
    # select columns from df with column names col_names
    exp_features = data_grouped[EXP_COL_NAMES]
    exp_data = exp_features.to_numpy().transpose()

    for col_data, col_name in zip(exp_data, exp_features.columns):
      # remove nan's from col_data
      data_corrected = col_data[~np.isnan(col_data)]
      if col_name == 'SpO2':
        data_corrected = 100 - data_corrected
      np.log(data_corrected, out=data_corrected, where=data_corrected > 0)
      # np.log10(data_corrected, out=data_corrected, where=data_corrected > 0)
      mean = np.mean(data_corrected)
      n, b, patches = plt.hist(data_corrected, range=(-5, 5), bins=30, density=True)
      max = np.argmax(n)
      # remove outliers for calculating std_deviation
      data_filtered = data_corrected[abs(data_corrected - mean) < 2*np.std(data_corrected)]
      scale = np.std(data_filtered)
      # take max occurence here instead of mean
      self.means[col_name] = max
      self.scales[col_name] = scale

  # apply transfom and scaling to all data except nan's
  # returns a dataframe with the transformed data
  def apply_scaler(self, data):
    # select columns from df with column names col_names
    norm_features = data[NORM_COL_NAMES]
    norm_data = norm_features.to_numpy().transpose()
    for col_data, col_name in zip(norm_data, norm_features.columns):
      # remove nan's from col_data
      data_corrected = col_data[~np.isnan(col_data)]
      if col_name == 'Fibrinogen' :
        data_corrected = np.log(data_corrected)
      if col_name == 'SaO2':
        data_corrected = 100 - data_corrected
        np.log10(data_corrected, out=data_corrected, where=data_corrected > 0)
      data_corrected -= self.means[col_name]
      data_corrected /= self.scales[col_name]
      col_data[~np.isnan(col_data)] = data_corrected
    # handle exponential distributions
    # select columns from df with column names col_names
    exp_features = data[EXP_COL_NAMES]
    exp_data = exp_features.to_numpy().transpose()
    for col_data, col_name in zip(exp_data, exp_features.columns):
      # remove nan's from col_data
      data_corrected = col_data[~np.isnan(col_data)]
      if col_name == 'SpO2':
        data_corrected = 100 - data_corrected
      np.log(data_corrected, out=data_corrected, where=data_corrected > 0)
      # np.log10(data_corrected, out=data_corrected, where=data_corrected > 0)
      data_corrected -= self.means[col_name]
      data_corrected /= self.scales[col_name]
      col_data[~np.isnan(col_data)] = data_corrected
    norm_df = pd.DataFrame(norm_data.transpose(), columns=NORM_COL_NAMES)
    exp_df = pd.DataFrame(exp_data.transpose(), columns=EXP_COL_NAMES)
    out_df = pd.concat([data['pid'], norm_df, exp_df], axis=1)
    return out_df

  # input scaled df of features
  # output is again one row per pid, with mean, trend and number of datapoints
  def prepoc_df(self, data):
    # add for each pid and each test -> trend, number of datapoints
    # for all patiens in data
    pids = data["pid"].unique()
    # output is again one row per pid, with mean, trend and number of datapoints
    out_df = data.groupby(['pid'], as_index=False).mean()
    # for each column add two additional columns _trend, _n_datapoints
    for col_name in data.columns[1:]:
      # get loc of col
      loc = out_df.columns.get_loc(col_name)
      out_df.insert(loc + 1, col_name + "_trend", 0, allow_duplicates = False)
      out_df.insert(loc + 2, col_name + "_n_datapoints", 0, allow_duplicates = False)
    # convert pid to index for faster write
    out_df.set_index('pid', inplace=True)
    with alive_bar(len(pids) * len(data.columns[1:])) as bar:
      for pid in pids:
        # for each test in data
        for test in data.columns[1:]:
          # get 12 datapoints for this test as a row
          test_data = data[data["pid"] == pid][test].to_numpy()
          # array of bool where is not nan
          is_not_nan_aray = ~np.isnan(test_data)
          # indices of not nan's
          not_nan_indices = np.argwhere(is_not_nan_aray).transpose()[0]
          # number of datapoints not nan
          n_data = len(not_nan_indices)
          out_df.at[pid, test + "_n_datapoints"] = n_data
          # out_df.loc[out_df["pid"] == pid, test + "_n_datapoints"] = n_data

          # replace nans with 0
          if n_data == 0 :
            out_df.at[pid, test] = 0
            # out_df.loc[out_df["pid"] == pid, test] = 0
          trend = 0
          if (n_data > 1):
            # calculate trend - fit line to data
            trend = np.polyfit(not_nan_indices, test_data[is_not_nan_aray], 1)[0]
            # out_df.at[pid, test + "_trend"] = trend
          # add trend to dataframe
          out_df.loc[pid, test + "_trend"] = trend
          bar()
    return out_df
  
  def preproc_regression(self, data):
    pids = data["pid"].unique()
    out_array = np.empty([0, 5])
    with alive_bar(len(pids) * len(VITAL_SIGNS)) as bar:
      for pid in pids:
        pid_array = np.empty([12, 5])
        pid_array[:, 0] = pid
        for idx, test in enumerate(VITAL_SIGNS):
          patient_data = data[data["pid"] == pid][test].to_numpy()
          if len(patient_data[~np.isnan(patient_data)]) == 0:
            patient_data = np.ones(12) * self.means[test]
          else: 
            mean = np.mean(patient_data[~np.isnan(patient_data)])
            patient_data[np.isnan(patient_data)] = mean
          pid_array[:, idx + 1] = patient_data # add column vector
          bar()
        out_array = np.append(out_array, pid_array, axis=0)
    out_df = pd.DataFrame(out_array, columns=['pid'] + VITAL_SIGNS)
    return out_df


if __name__ == "__main__":
  DataProcessor()
