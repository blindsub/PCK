import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch.optim
# import dnn-mmd

class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_out):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(n_feature, n_hidden)
        self.hidden2 = nn.Linear(n_hidden, n_hidden)
        self.hidden3 = nn.Linear(n_hidden, n_hidden)
        self.hidden4 = nn.Linear(n_hidden, n_hidden)
        self.hidden5 = nn.Linear(n_hidden, n_hidden)
        self.hidden6 = nn.Linear(n_hidden, n_hidden)

        self.predict = nn.Linear(n_hidden, n_out)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.hidden4(x))
        x = F.relu(self.hidden5(x))
        x = F.relu(self.hidden6(x))

        out = self.predict(x)

        return out

