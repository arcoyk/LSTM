from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
D = 5

# model = LSTMCell.train(X, y)
# model.predict(X) # -> y

def learn(py):
  lstm = nn.LSTMCell(10, 10)
  matrix = py.reshape(10, 10)
  tensor = torch.from_numpy(matrix)
  input = Variable(tensor.double())
  # print(input[0])
  # input = Variable(torch.randn(6, 3, 10))
  hidden = Variable(torch.randn(3, 10).double())
  cell = Variable(torch.randn(3, 10).double())
  output = []
  for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
    hidden, cell = lstm(input_t, (hidden, cell))
    output.append(hidden)
  return output

def predict(px):
  return np.sin(px / D)

def run():
  x = np.array(range(100))
  y = np.sin(x / D)
  px = x[-1] + np.array(range(100))
  py = predict(px)
# MAT PLOT
  plt.scatter(x, y)
  plt.scatter(px, py, c='r')
  plt.show()

x = np.array(range(100))
learn(x)
