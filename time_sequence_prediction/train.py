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
        self.L = 100
        self.lstm1 = nn.LSTMCell(1, self.L)
        self.lstm2 = nn.LSTMCell(self.L, 1)

    def forward(self, input, future = 0):
        outputs = []
        h_t = Variable(torch.zeros(input.size(0), self.L).double(), requires_grad=False)
        c_t = Variable(torch.zeros(input.size(0), self.L).double(), requires_grad=False)
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

if __name__ == '__main__':
    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)
    # load data and make training set
    data = torch.load('traindata.pt')
    # Each element of data is a fraction of sin wave
    # data = [[0.2, 0.1 ... 0.2], [0.2, 0.4 ... 0.23] ... [0.1, 0.3 ... 0.5]]
    # 0.1, 0.2, 0.4, 0.1, 0.3, 0.5
    # <------- input ------->
    #      <------ target ------->
    LL = 50
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
        # y = pred.data.numpy()
        # draw the result
        # plt.plot(input.data[i].numpy())
        # plt.plot(pred.data[3].numpy())
        for m in range(10):
            past = test_input.data[m].numpy()
            post = pred.data[m].numpy()
            past = list(past) + list(post)
            plt.plot(past)
        plt.show()
        plt.savefig('predict%d.pdf'%i)
        plt.close()
