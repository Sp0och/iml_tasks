from numpy import dtype
import pandas as pd
import numpy as np
from numpy.linalg import lstsq

test_file = "./train.csv"

data = pd.read_csv(test_file, float_precision="round_trip")
data = data.to_numpy()
y = np.reshape(data[:,1], (-1, 1))
x = data[:,2:]

x2 = np.square(x)
exponential = np.exp(x)
cos = np.cos(x)

# least squares fit
A = np.concatenate([x, x2, exponential, cos, np.ones_like(y)], axis=1)
sol = lstsq(A, y, rcond=None)[0]

print(sol)
np.savetxt("results.csv", sol, delimiter="\n", fmt='%4.12f')