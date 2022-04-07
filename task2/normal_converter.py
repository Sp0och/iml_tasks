import pandas as pd
import numpy as np
from scipy.stats import norm, skew
import matplotlib.pyplot as plt

# compress all patient datapoints along time axis, taking the newest time if
# multiple datapoints are available for that pid
df = pd.read_csv("./train_features.csv")
# group by pid and take row with newest time TODO take latest datapoint
df = df.groupby(['pid']).mean()

norm_col_names = ['SaO2', 'Fibrinogen', 'EtCO2', 'Temp', 'Hgb', 'HCO3', 
    'BaseExcess', 'RRate', 'Phosphate', 'PaCO2', 'Platelets', 'Glucose', 'ABPm',
    'Magnesium', 'Potassium', 'ABPd', 'Chloride', 'Hct', 'Heartrate', 'ABPs',
    'pH', 'WBC', 'FiO2']
# select columns from df with column names col_names
norm_features = df[norm_col_names]

# read colums with col_names from csv
norm_data = norm_features.to_numpy().transpose()

for col_data, col_name in zip(norm_data, norm_features.columns):
  # remove nan's from col_data
  data_corrected = col_data[~np.isnan(col_data)]
  if col_name == 'Fibrinogen' :
    data_corrected = np.log(data_corrected)
  if col_name == 'SaO2':
    # print(data_corrected)
    data_corrected = 100 - data_corrected[data_corrected < 100]
    data_corrected = np.log10(data_corrected)
  mean = np.mean(data_corrected)
  # remove outliers for calculating std_deviation
  data_corrected = data_corrected[abs(data_corrected - mean) < 2*np.std(data_corrected)]
  mean = np.mean(data_corrected)
  std_deviation = np.std(data_corrected)
  # replace elements in col_data where nan with mean
  if col_name == 'Fibrinogen':
    col_data = np.log(col_data)
  if col_name == 'SaO2':
    col_data = 100 - col_data[col_data < 100]
    col_data = np.log10(col_data)
  col_data -= mean
  col_data = col_data / std_deviation
  plot_data = col_data[~np.isnan(col_data)]
  col_data[np.isnan(col_data)] = mean
  # plot mean and standart deviation
  plt.hist(plot_data, range=(-5, 5), bins=20, label=col_name, density=True, stacked=True)
  x_axis = np.linspace(-5, 5, 100)
  plt.plot(x_axis, norm.pdf(x_axis, 0, 1), label="normal distribution")
  plt.legend()
  plt.show()

# skew_col_names = ['Fibrinogen', 'SaO2']
# # select columns from df with column names col_names
# skew_features = df[skew_col_names]

# # # read colums with col_names from csv
# skew_data = skew_features.to_numpy().transpose()

# for col_data, col_name in zip(skew_data, skew_features.columns):
#   # remove nan's from col_data
#   data_corrected = col_data[~np.isnan(col_data)]
#   # data_corrected = np.log(data_corrected)
#   mean = np.mean(data_corrected)
#   # remove outliers for calculating std_deviation
#   data_corrected = data_corrected[abs(data_corrected - mean) < 2*np.std(data_corrected)]
#   mean = np.mean(data_corrected)
#   std_deviation = np.std(data_corrected)
#   # replace elements in col_data where nan with mean
#   col_data -= mean
#   col_data = col_data / std_deviation
#   plot_data = col_data[~np.isnan(col_data)]
#   col_data[np.isnan(col_data)] = mean
#   # plot mean and standart deviation
#   plt.hist(plot_data, range=(-5, 5), bins=20, label=col_name, density=True, stacked=True)
#   x_axis = np.linspace(-5, 5, 100)
#   plt.plot(x_axis, norm.pdf(x_axis, 0, 1), label="normal distribution")
#   plt.legend()
#   plt.show()



