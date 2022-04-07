import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

features = pd.read_csv("./train_features.csv")
labels = pd.read_csv("./train_labels.csv")
data = features.to_numpy().transpose()

#plot histogramm of columns of labels
for col_data, col_name in zip(data, features.columns):
    plt.hist(col_data, bins=100, label=col_name)
    plt.legend()
    plt.show()
    # save plot in plots folder
    plt.savefig(f"./plots/{col_name}.png")