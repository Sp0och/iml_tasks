import torch
import pandas as pd
import numpy as np

data = pd.read_csv("train.csv")
labels = data.iloc[2][1]
datapoints = data.iloc[2][2:].to_numpy()
mean = datapoints.mean()
MSE = np.sqrt(np.sum(np.square((datapoints-mean)))/10)
print(f"Datapoints: {datapoints}, \n mean: {mean}, MSE = {MSE}, y = {labels}")
