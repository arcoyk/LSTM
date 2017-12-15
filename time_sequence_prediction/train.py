from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        # LSTMCell(input_size, hidden_size)
        # Why?
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
        # for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
        # for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            # input_t.data.shape => [19 x 1 x 2]
            input_t = torch.squeeze(input_t)
            # input_t.data.shape => [19 x 2]
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            outputs += [h_t2]
        for i in range(future):# if we should predict the future
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

if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    data = traj_load('traj.pt')
    LL = int(len(data) * 0.8)
    input = Variable(torch.from_numpy(data[:LL, :-1]), requires_grad=False)
    target = Variable(torch.from_numpy(data[:LL, 1:]), requires_grad=False)
    test_input = Variable(torch.from_numpy(data[LL:, :-1]), requires_grad=False)
    test_target = Variable(torch.from_numpy(data[LL:, 1:]), requires_grad=False)
    # build the model
    # This could be RNN
    seq = Sequence()
    # nn.Module (parent class) .double() convert float into double
    seq.double()
    # Loss function: loss = criterion(nn.output, target)
    # MSE stands for Mean Square Error
    criterion = nn.MSELoss()
    # use pytorch.optim.LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.LBFGS(seq.parameters(), lr=0.8)
    #begin to train
    # 15 iterations
    for i in range(15):
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
        # plt.show()
        plt.close()
        # exit()
