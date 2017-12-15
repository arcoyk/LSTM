from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        # LSTMCell(input_size, hidden_size)
        self.L = 10
        self.IN = 2
        self.OUT = 2
        self.lstm1 = nn.LSTMCell(self.IN, self.L)
        self.lstm2 = nn.LSTMCell(self.L, self.OUT)

    def forward(self, input, future = 0):
        # input.data.shape => [19 x 99 x 21]
        # input.size(0) => 19
        # input.size(1) => 99
        outputs = []
        h_t = Variable(torch.zeros(input.size(0), self.L).double(), requires_grad=False)
        c_t = Variable(torch.zeros(input.size(0), self.L).double(), requires_grad=False)
        h_t2 = Variable(torch.zeros(input.size(0), self.OUT).double(), requires_grad=False)
        c_t2 = Variable(torch.zeros(input.size(0), self.OUT).double(), requires_grad=False)
        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            # input_t.data.shape => [19 x 1 x 2]
            input_t = torch.squeeze(input_t)
            # input_t.data.shape => [19 x 2]
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            outputs += [h_t2]
        for i in range(future):
            h_t, c_t = self.lstm1(h_t2, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            outputs += [h_t2]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs

def traj_load(file_name):
    rst = []
    mini_bach = []
    mini_bach_size = 100
    with open(file_name) as f:
        for line in f.read().split('\n'):
            nums = line.split(' ')
            if len(nums) == 2:
                nums = list(map(lambda x:float(x), nums))
                mini_bach.append(nums)
                if len(mini_bach) == mini_bach_size:
                    rst.append(mini_bach)
                    mini_bach = []
    rst = np.array(rst)
    return rst

def plot_line(line, c_in='r', m_in='x'):
    for p in line:
        plt.scatter(p[0], p[1], c=c_in, marker=m_in)

# build the model
# This could be RNN
# nn.Module (parent class) .double() convert float into double
# Loss function: loss = criterion(nn.output, target)
# use pytorch.optim.LBFGS as optimizer since we can load the whole data to train
# MSE stands for Mean Square Error
seq = Sequence()
seq.double()
criterion = nn.MSELoss()
optimizer = optim.LBFGS(seq.parameters(), lr=0.8)
np.random.seed(0)
torch.manual_seed(0)
SRC_FILE = 'teach.txt'
IN_FILE = 'in.txt'
OUT_FILE = 'out.txt'

def learn():
    data = traj_load(SRC_FILE)
    LL = int(len(data) * 0.8)
    input, target = input_target(data)
    input = Variable(torch.from_numpy(input[:LL]), requires_grad=False)
    target = Variable(torch.from_numpy(target[:LL]), requires_grad=False)
    test_input = Variable(torch.from_numpy(input[LL:]), requires_grad=False)
    test_target = Variable(torch.from_numpy(target[LL:]), requires_grad=False)
    for i in range(3):
        print('STEP: ', i)
        def closure():
            optimizer.zero_grad()
            out = seq(input)
            loss = criterion(out, target)
            print('loss:', loss.data.numpy()[0])
            loss.backward()
            return loss
        optimizer.step(closure)
        # begin to predict
        future = 100
        pred = seq(test_input, future = future)
        loss = criterion(pred[:, :-future], test_target)
        print('test loss:', loss.data.numpy()[0])
        for line in input.data:
            plot_line(line, c_in='b')
        for line in pred.data:
            past = line[:-future]
            post = line[-future:]
            plot_line(past, c_in='g')
            plot_line(post, c_in='r')
        plt.savefig('predict%d.pdf'%i)
        plt.close()

def predict(input):
    return seq(input)

in_file_timestamp = 0
def file_updated():
    global in_file_timestamp
    curr = os.path.getmtime(IN_FILE)
    if in_file_timestamp != curr:
        in_file_timestamp = curr
        return True
    return False

DELIM = ':'
def last_line(file_name):
    with open(file_name) as f:
        lines = f.read().split('\n')
        for i in range(len(lines)):
            line = lines[-(i+1)]
            if DELIM in line:
                return line

def push_last_line(file_name, s):
    with open(file_name, 'a') as f:
        f.write('\n')
        f.write(s)

def input_target(line):
    input, target = line.split(':')
    input = list(map(lambda x:float(x), input.split(',')))
    if target != '':
        target = list(map(lambda x:float(x), target.split(',')))
    return input, target

def join_input_target(input, target):
    input = ','.join(map(str, input))
    target = ','.join(map(str, target))
    return input + ':' + target

def watch():
    while True:
        if file_updated():
            line = last_line(IN_FILE)
            input, target = input_target(line)
            if len(target) < 2:
                # out = predict(input)
                out = np.array([1.23, 1.64])
                s = join_input_target(input, out)
                push_last_line(OUT_FILE, s)
            else:
                push_last_line(SRC_FILE, line)
                # learn()
        exit()

watch()

