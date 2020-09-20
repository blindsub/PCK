import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.optim
from modle import Net
import time

print('Read whole dataset from csv file ...')
dir_data = '/Users/weishouxin/Documents/Program/Python/database-transfer/data/mysql-0905/mysql_5.5.csv'
print('Dataset: ' + dir_data)
whole_data = np.genfromtxt(dir_data, delimiter=',', skip_header=1)
(N, n) = whole_data.shape
n = n - 1

X_all = whole_data[:, :n]
Y_all = whole_data[:, n][:, np.newaxis]

max_X_all = np.amax(X_all, axis=0)
if 0 in max_X_all:
    max_X_all[max_X_all == 0] = 1
X_all = np.divide(X_all, max_X_all)

max_Y_all = np.max(Y_all)/100
if max_Y_all == 0:
    max_Y_all = 1
Y_all = np.divide(Y_all, max_Y_all)

x_all = torch.FloatTensor(X_all)
y_all = torch.FloatTensor(Y_all)
x_all = Variable(x_all)
y_all = Variable(y_all)

time1 = time.time()

net = Net(n_feature=n, n_hidden=128, n_out=1)
loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.01)

epochs = 300
for i in range(epochs):
    prediction = net(x_all)
    loss = loss_func(prediction, y_all)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(i+1, "loss:", loss.data.item())

torch.save(net.state_dict(), "models/my_model.pkl")

time2 = time.time()
print((time2-time1)*1000)

