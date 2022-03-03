import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, annotations_file):
        self.data = pd.read_csv(annotations_file)
        self.labels = self.data['y']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        datapoints = row[2:].to_numpy()
        return datapoints, self.labels.iloc[idx]