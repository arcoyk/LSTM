# Generate 1000 sin waves, started with different point

import numpy as np
import torch

np.random.seed(2)

T = 20 # Randomness
L = 1000 # Number of columns
N = 100 # Number of rows

x = np.empty((N, L), 'int64')
# np.array(range(5)) => array([0, 1, 2, 3, 4])
# randint(2, 4, 5) => array([2, 3, 3, 2, 3])
# array([1,2,3,4,5,6]).reshape(2, 3) => array([[1, 2, 3], [4, 5, 6]])
# Shift [1,2,3,4...L] to random(-20, 20)
x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
# Convert each column to sin
data = np.sin(x / 1.0 / T).astype('float64')
torch.save(data, open('traindata.pt', 'wb'))
