import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from alive_progress import alive_bar

# handle normal distributions
NORM_COL_NAMES = ['SaO2', 'Fibrinogen', 'EtCO2', 'Temp', 'Hgb', 'HCO3', 
'BaseExcess', 'RRate', 'Phosphate', 'PaCO2', 'Platelets', 'Glucose', 'ABPm',
'Magnesium', 'Potassium', 'ABPd', 'Chloride', 'Hct', 'Heartrate', 'ABPs',
'pH', 'FiO2']

EXP_COL_NAMES = ['PTT', 'BUN', 'Lactate', 'WBC', 'Creatinine', 'AST', 
    'Alkalinephos', 'SpO2', 'Bilirubin_direct', 'Bilirubin_total',
    'TroponinI']

def import_from_file(filename):
  data = pd.read_csv("./input_data/" + filename + ".csv")
  return data

# for each feature we need to find 
# - replacement value for missing entries: mean or max occurence
# - scaling factor: std
# - transform function: (log)(x - mean)/std 
# then we can first look at the raw data
# if there is one measurement for the patient -> take that one everywhere
# if there are multiple measurements for the patient -> inter/extrapolate them linearly
# if there are no measurements leave a nan
# then transform the whole data

# find means of data

class DataProcessor:
  def __init__(self):
    train_features_df = import_from_file("train_features")
    self.means = {}
    self.scales = {}
    print("[Starting] Finding scales and means")
    self.train_scaler(train_features_df)
    print("[Finished] Finding scales and means")
    print("[Starting] Appling feature transformations to train data")
    scaled_train_features = self.apply_scaler(train_features_df)
    print("[Finished] Appling feature transformations to train data")
    print("[Starting] Appling feature transformations to test data")
    scaled_test_features = self.apply_scaler(import_from_file("test_features"))
    print("[Finished] Appling feature transformations to test data")
    # test on partial dataset
    processed_train_df = self.prepoc_df(scaled_train_features.iloc[0:1000, :])
    print("[Starting] Preprocessing normalized train features")
    # processed_train_df = self.prepoc_df(scaled_train_features)
    processed_train_df.to_csv("./output_data/train_features_processed.csv", index=False)
    print("[Finished] Preprocessing normalized train features")
    print("[Starting] Preprocessing normalized test features")
    processed_test_df = self.prepoc_df(scaled_test_features)
    processed_test_df.to_csv("./output_data/test_features_processed.csv", index=False)
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
    with alive_bar(len(pids) * len(data.columns[1:])) as bar:
      for pid in pids:
        # for each test in data
        for test in data.columns[1:]:
          # get 12 datapoints for this test as a row
          test_data = data[data["pid"] == pid][test].values
          # indices of not nan's
          not_nan_indices = np.argwhere(~np.isnan(data[data["pid"] == pid][test].values)).transpose()[0]
          # number of datapoints not nan
          n_data = np.count_nonzero(~np.isnan(test_data))
          out_df.loc[out_df["pid"] == pid, test + "_n_datapoints"] = n_data

          # replace nans with 0
          if n_data == 0 :
             out_df.loc[out_df["pid"] == pid, test] = 0
          
          trend = 0
          if (n_data > 1):
            # calculate trend - fit line to data
            trend = np.polyfit(not_nan_indices, test_data[~np.isnan(test_data)], 1)[0]
            # due to prior normalization, trend is also normalized
          # add trend to dataframe
          out_df.loc[out_df["pid"] == pid, test + "_trend"] = trend
          bar()
    return out_df
          
        
          
    #     # if test is not nan
    #     if not np.isnan(data[data["pid"] == pid][test].values[0]):
    #       # get all values for test
    #       values = data[data["pid"] == pid][test].values
    #       # get all values for test that are not nan
    #       values_corrected = values[~np.isnan(values)]
    #       # get all values for test that are not nan and are not zero
    #       values_corrected_nonzero = values_corrected[values_corrected != 0]
    #       # get all values for test that are not nan and are not zero and are not one
    #       values_corrected_nonzero_nonone = values_corrected_nonzero[values_corrected_nonzero != 1]
    #       # get all values for test that are not nan and are not zero and are not one and are not two
    #       values_corrected_nonzero_nonone_nontwo = values_corrected_nonzero_nonone[values_corrected_nonzero_nonone != 2]
    #       # get all values for test that are not nan and are not zero and are not one and are not two and are not three
    #       values_corrected_nonzero_nonone_nontwo_nonthree = values_corrected_nonzero_nonone_nontwo[values_corrected_nonzero_nonone_nontwo != 3]
    #       # get all values for test that are not nan and are not zero and are not one and are not two and are not three and are not four
    #       values_corrected_nonzero_nonone_nontwo_nonthree_nonfour = values_corrected_nonzero_nonone_nontwo_nonthree[values_corrected_nonzero_nonone_nontwo_nonthree != 4]
    #       # get all values for test that are not nan and are not zero and are not one and are not two and are not three and are not four and are not five
    #       values_corrected_nonzero_nonone_nontwo_nonthree_nonfour_nonfive = values_corrected_nonzero_

if __name__ == "__main__":
  DataProcessor()
