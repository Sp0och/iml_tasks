from numpy import dtype
from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np

# split train dataset into 10 equal parts
test_file = "./train.csv"

data = pd.read_csv(test_file).to_numpy()
# print(data)
folds = np.split(data,[15,30,45,60,75,90,105,120,135], axis=0)

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

# np.set_printoptions(suppress=True)
np.savetxt("results.csv", average_RMSEs, delimiter="\n", fmt='%10.12f')
print(average_RMSEs)