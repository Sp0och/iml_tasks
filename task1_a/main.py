from numpy import dtype
from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np

# split train dataset into 10 equal parts
test_file = "./train.csv"

data_unshuffled = pd.read_csv(test_file).to_numpy()
data = data_unshuffled

fold_size = int(len(data)/10)
split_idx = np.linspace(fold_size, 9*fold_size, 9).astype(int)

array_average_RMSEs = []
for seed_ in range(1,10000,50):
    data = data_unshuffled
    np.random.seed(seed_)
    np.random.shuffle(data)
    folds = np.split(data, split_idx, axis=0)

    # repeat for different lambda
    alphas = [0.1, 1, 10, 100, 200]
    average_RMSEs = []
    for alpha_ in alphas:
        RMSEs = []
        # repeat leaving out 1 different fold each time:
        for i in range (0, 10):
            # find weights with 9 parts of train data
            used_folds = np.delete(folds, i, axis=0)
            y = np.concatenate(used_folds[:, :, 0], axis=0)
            x = np.concatenate(used_folds[:, :, 1:], axis = 0)
            estimator = Ridge(alpha=alpha_)
            estimate = estimator.fit(x, y)
        
            # find RMSE of remaining part
            validation_fold = folds[i]
            n = len(validation_fold)
            x = validation_fold[:, 1:]
            y = validation_fold[:, 0]
            y_hat = estimator.predict(x)
            rmse = np.sqrt(1/n * np.sum( np.square(y - y_hat)))
            RMSEs.append(rmse)
        # average over all the RMSE's
        average_RMSE = 1 / len(RMSEs) * np.sum(RMSEs)
        average_RMSEs.append(average_RMSE)
    array_average_RMSEs.append(average_RMSEs)
average_average_RMSEs = 1 / len(array_average_RMSEs) * np.sum(array_average_RMSEs, axis=0)

# np.set_printoptions(suppress=True)
np.savetxt("results.csv", average_average_RMSEs, delimiter="\n", fmt='%10.12f')
print(average_average_RMSEs)