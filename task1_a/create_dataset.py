import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class CustomTrainDataset(Dataset):
    def __init__(self, annotations_file):
        self.data = pd.read_csv(annotations_file)
        self.labels = self.data['y']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        datapoints = row[2:].to_numpy()
        return datapoints, self.labels.iloc[idx]

class CustomTestDataset(Dataset):
    def __init__(self, validation_file):
        self.data = pd.read_csv(validation_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        datapoints = row[1:].to_numpy()
        index = row[0:1].to_numpy()
        return datapoints, index