import pandas as pd
import numpy as np

features = pd.read_csv("train_features.csv")
labels = pd.read_csv("train_labels.csv")
right_part = features.to_numpy()
# for column in range(3,len(right_part.T),1):
#     # print(f"column: {column}")
#     c0 = np.nan_to_num(right_part.T[column])
#     average = np.mean(c0)
#     right_part.T[column] = np.nan_to_num(right_part.T[column],nan=average)
#     # print(f"column: {right_part.T[column]}")
# right_part = right_part.round(5)
# # print(f"right part: {right_part}")
# np.savetxt("averages.csv",right_part,delimiter=",",fmt="%1.4f")

#todo: implement the same but consider average built from the healthy patients:
for column in range(3,len(right_part.T),1):
    # print(f"column: {column}")
    c0 = np.nan_to_num(right_part.T[column])
    average = np.mean(c0)
    right_part.T[column] = np.nan_to_num(right_part.T[column],nan=average)
    # print(f"column: {right_part.T[column]}")
right_part = right_part.round(5)
# print(f"right part: {right_part}")
np.savetxt("averages.csv",right_part,delimiter=",",fmt="%1.4f")