import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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
    self.scaler = StandardScaler()
    self.means = {}
    self.scales = {}
    self.train_scaler(train_features_df)
    print(self.means['Chloride'])
    self.apply_scaler(train_features_df)
    self.prepoc_df(train_features_df)

  # returns a dict of trained scalers
  def train_scaler(self, data):
    data = data.groupby(['pid'], as_index=False).mean()

    # handle normal distributions
    norm_col_names = ['SaO2', 'Fibrinogen', 'EtCO2', 'Temp', 'Hgb', 'HCO3', 
    'BaseExcess', 'RRate', 'Phosphate', 'PaCO2', 'Platelets', 'Glucose', 'ABPm',
    'Magnesium', 'Potassium', 'ABPd', 'Chloride', 'Hct', 'Heartrate', 'ABPs',
    'pH', 'WBC', 'FiO2']

    # select columns from df with column names col_names
    norm_features = data[norm_col_names]

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
    exp_col_names = ['PTT', 'BUN', 'Lactate', 'WBC', 'Creatinine', 'AST', 
    'Alkalinephos', 'SpO2', 'Bilirubin_direct', 'Bilirubin_total',
    'TroponinI']

    # select columns from df with column names col_names
    exp_features = data[exp_col_names]

    # read colums with col_names from csv
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

  # nan's are maintained in transform
  def apply_scaler(self, data):
    # disregard first three columns (pid, time, age)
    columns = data.to_numpy()
    columns = columns[3:]
    self.scaler.transform(columns)

  def prepoc_df(self, data):
    # apply scaler to all columns

    # for all patiens in data
    pids = data["pid"].unique()

if __name__ == "__main__":
  DataProcessor()
