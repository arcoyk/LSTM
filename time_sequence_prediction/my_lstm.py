import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import random

class Sequence(nn.Module):
  def __init__(self):
    super(Sequence, self).__init__()
    # LSTMCell(input_size, hidden_size)
    # Why?
    self.lstm1 = nn.LSTMCell(1, 5)
    self.lstm2 = nn.LSTMCell(5, 1)

  def forward(self, input, future = 0):
    outputs = []
    h_t = Variable(torch.zeros(input.size(0), 5).double(), requires_grad=False)
    c_t = Variable(torch.zeros(input.size(0), 5).double(), requires_grad=False)
    h_t2 = Variable(torch.zeros(input.size(0), 1).double(), requires_grad=False)
    c_t2 = Variable(torch.zeros(input.size(0), 1).double(), requires_grad=False)
    for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
      # 3 input and 2 output
      h_t, c_t = self.lstm1(input_t, (h_t, c_t))
      h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
      outputs += [h_t2]
    for i in range(future):# if we should predict the future
      h_t, c_t = self.lstm1(h_t2, (h_t, c_t))
      h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
      outputs += [h_t2]
    outputs = torch.stack(outputs, 1).squeeze(2)
    return outputs

# data = [[1, 2, 3, 4, 5], [4, 5, 6, 7, 8] ... [34, 35, 36, 37, 38]]
data = []
for i in range(10):
  r = int(random.random() * 100)
  data.append([r + m for m in range(6)])
data = np.array(data)

input = Variable(torch.from_numpy(data[3:, :-1]))
print(input[0])
target = Variable(torch.from_numpy(data[3:, -1:]))
test_input = Variable(torch.from_numpy(data[:3, :-1]))
test_target = Variable(torch.from_numpy(data[:3, -1:]))

seq = Sequence()
seq.double()
cri = nn.MSELoss()
# opt = optim.LBFGS(seq.parameters(), lr = 0.8)
opt = optim.Adam(seq.parameters(), lr = 0.8)

for i, m in enumerate(seq.modules()):
  print(i, '->', m)

for i in range(15):
  # print('Step: ', i)
  opt.zero_grad()
  out = seq(input)
  # loss = cri(out, target)
  # print('loss:', loss.data.numpy()[0])
  # loss.backward()
  # opt.step(closure)
