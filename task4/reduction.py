import numpy as np
import pandas as pd
from sklearn import preprocessing
import sklearn.decomposition as pca
import matplotlib.pyplot as plt

#TODO load in datasets

pretrain_data = pd.read_csv('./pretrain_features.csv.zip')
pretrain_start = pretrain_data[['Id','smiles']].to_numpy()
pretrain_trimmed = pretrain_data.drop(['Id','smiles'], axis=1)

train_data = pd.read_csv('./train_features.csv.zip')
train_start = train_data[['Id','smiles']].to_numpy()
train_trimmed = train_data.drop(['Id','smiles'], axis=1)

test_data = pd.read_csv('./test_features.csv.zip')
test_start = test_data[['Id','smiles']].to_numpy()
test_trimmed = test_data.drop(['Id','smiles'], axis=1)

#TODO Preprocess data to achieve mean = 0 and var = 1 NOTE scaler trained on pretrain features but applied to all three

scaler = preprocessing.StandardScaler().fit(pretrain_trimmed)
pretrain_data_processed = scaler.transform(pretrain_trimmed)
train_data_processed = scaler.transform(train_trimmed)
test_data_processed = scaler.transform(test_trimmed)
# print(f"mean of scaler: {data_processed.mean(axis=0)}")
# print(f"size of scaler: {data_processed.std(axis=0)}")

#TODO Apply dimensionality reduction to break the 1000 features down into principle components NOTE not sure if same pca is good procedure but I think the more similar the NN input the easier to apply transfer learning

pca_handler = pca.PCA(n_components=10)  # TODO see with how many features per molecule we perform best 
pca_handler.fit(pretrain_data_processed)
pca_pretrain_data = pca_handler.transform(pretrain_data_processed)
pca_train_data = pca_handler.transform(train_data_processed)
pca_test_data = pca_handler.transform(test_data_processed)
print(f"type of pca_datasets: {type(pca_train_data)}")

#TODO save new datasets

feature_cols = ['feature' + str(i) for i in range(pca_train_data.shape[1])]
processed_pretrain_dataset = pd.DataFrame(pca_pretrain_data,columns=feature_cols)
processed_pretrain_dataset.insert(0,['Id','smiles'],pretrain_start)
processed_train_dataset = pd.DataFrame(pca_train_data,columns=feature_cols)
processed_train_dataset.insert(0,['Id','smiles'],train_start)
processed_test_dataset = pd.DataFrame(pca_test_data,columns=feature_cols)
processed_test_dataset.insert(0,['Id','smiles'],test_start)

print(f"pretrain dataset: {processed_pretrain_dataset}")
print(f"train dataset: {processed_train_dataset}")
print(f"test dataset: {processed_test_dataset}")


#NOTE If we want we can extract more features from the chemistry database to enhance input data