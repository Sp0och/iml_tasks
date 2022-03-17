import torch
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
from create_dataset import CustomTestDataset

def predict(validation_file, model):

  # put the model into evaluation mode
  model.eval

  test_dataset = CustomTestDataset(validation_file=validation_file)

  # load the data in the right order for submission
  test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=False)

  def evaluate_loop(dataloader, model):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    result_df = pd.DataFrame(columns = ["Id", "y"])
    with torch.no_grad():
        for X, idx in dataloader:
            pred = model(X)
            # torch.tensor.numpy() converts to numpy array
            data = np.concatenate((idx.numpy().astype(int), pred.numpy()), axis = 1, dtype=str)
            result_df = pd.concat([result_df, pd.DataFrame(data, columns = ["Id", "y"])], axis = 0)
    result_df.to_csv("results.csv", index=False)

  evaluate_loop(test_dataloader, model)

  print("Done predicting!")

