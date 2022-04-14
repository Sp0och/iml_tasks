import pandas as pd
import numpy as np

features = pd.read_csv("train_features.csv")
labels = pd.read_csv("train_labels.csv").to_numpy()
rel_labels = labels[:,1:12]

healthy_features = features[np.all(rel_labels == 0,axis=1)]
print(f"hf shape: {healthy_features.shape}")

# derive the distribution summarizing equivalent for the healthy features and fill the nans with it to fill it with healthy data that surely doesn't provoke tests / Sepsis 

# right_part = features.to_numpy()
# # for column in range(3,len(right_part.T),1):
# #     # print(f"column: {column}")
# #     c0 = np.nan_to_num(right_part.T[column])
# #     average = np.mean(c0)
# #     right_part.T[column] = np.nan_to_num(right_part.T[column],nan=average)
# #     # print(f"column: {right_part.T[column]}")
# # right_part = right_part.round(5)
# # # print(f"right part: {right_part}")
# # np.savetxt("averages.csv",right_part,delimiter=",",fmt="%1.4f")

# #todo: implement the same but consider average built from the healthy patients:
# for column in range(3,len(right_part.T),1):
#     # print(f"column: {column}")
#     c0 = np.nan_to_num(right_part.T[column])
#     average = np.mean(c0)
#     right_part.T[column] = np.nan_to_num(right_part.T[column],nan=average)
#     # print(f"column: {right_part.T[column]}")
# right_part = right_part.round(5)
# # print(f"right part: {right_part}")
# np.savetxt("averages.csv",right_part,delimiter=",",fmt="%1.4f"