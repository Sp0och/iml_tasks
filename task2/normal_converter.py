import pandas as pd
import numpy as np
from scipy.stats import norm, skew
import matplotlib.pyplot as plt

# compress all patient datapoints along time axis, taking the newest time if
# multiple datapoints are available for that pid
df = pd.read_csv("./train_features.csv")
# group by pid and take row with newest time TODO take latest datapoint
df = df.groupby(['pid'], as_index=False).mean()

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
    data_corrected = 100 - data_corrected
    np.log10(data_corrected, out=data_corrected, where=data_corrected > 0)
  mean = np.mean(data_corrected)
  # remove outliers for calculating std_deviation
  data_filtered = data_corrected[abs(data_corrected - mean) < 2*np.std(data_corrected)]
  mean = np.mean(data_filtered)
  std_deviation = np.std(data_filtered)
  # correct the data with mean and std
  data_corrected -= mean
  data_corrected /= std_deviation
  col_data[~np.isnan(col_data)] = data_corrected
  # replace elements in col_data where nan with 0 (mean of normalized distribution)
  col_data[np.isnan(col_data)] = 0
  # plot mean and standart deviation
  # plt.hist(data_corrected, range=(-5, 5), bins=20, label=col_name, density=True, stacked=True)
  # x_axis = np.linspace(-5, 5, 100)
  # plt.plot(x_axis, norm.pdf(x_axis, 0, 1), label="normal distribution")
  # plt.legend()
  # plt.show()


## -- normally distributed features are correctly transformed in 
## -- in norm_data                                         


exp_col_names = ['PTT', 'BUN', 'Lactate', 'WBC', 'Creatinine', 'AST', 
    'Alkalinephos', 'SpO2', 'Bilirubin_direct', 'Bilirubin_total',
    'TroponinI']

# select columns from df with column names col_names
exp_features = df[exp_col_names]

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
  # remove outliers for calculating std_deviation
  data_filtered = data_corrected[abs(data_corrected - mean) < 2*np.std(data_corrected)]
  mean = np.mean(data_filtered)
  std_deviation = np.std(data_filtered)
  # correct the data with mean and std
  data_corrected -= mean
  data_corrected /= std_deviation
  col_data[~np.isnan(col_data)] = data_corrected
  n, b, patches = plt.hist(data_corrected, range=(-5, 5), bins=30, density=True)
  max = np.argmax(n)
  # replace elements in col_data where nan with the maximum of the other data
  col_data[np.isnan(col_data)] = max
  # plot mean and standart deviation
  # plt.hist(col_data, range=(-5, 5), bins=20, label=col_name, density=True, stacked=True)
  # x_axis = np.linspace(-5, 5, 100)
  # plt.plot(x_axis, norm.pdf(x_axis, 0, 1), label="normal distribution")
  # plt.legend()
  # plt.show()

## -- normally distributed features are correctly transformed in 
## -- in exp_data     


# store all of it in a csv file
norm_df = pd.DataFrame(norm_data.transpose(), columns=norm_col_names)
exp_df = pd.DataFrame(exp_data.transpose(), columns=exp_col_names)
out_df = pd.concat([df['pid'], norm_df, exp_df], axis=1)
out_df.to_csv('out.csv', index=False)  