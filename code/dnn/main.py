import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.optim
from modle import Net
import pandas as pd
import time


# dir_data1 = 'data/mysqlResult_8.0.csv'
# print('Dataset: ' + dir_data1)
# whole_data1 = np.genfromtxt(dir_data1, delimiter=',', skip_header=1)
# (N1, n1) = whole_data1.shape
# n1 = n1 - 1
#
# N_train1 = 150
# permutation1 = np.random.permutation(N1)
# training_index1 = permutation1[0:N_train1]
# training_data1 = whole_data1[training_index1, :]
# X_train1 = training_data1[:, 0:n1]
# Y_train1 = training_data1[:, n1][:, np.newaxis]
#
# X_all = whole_data1[:, :n1]
# Y_all = whole_data1[:, n1][:, np.newaxis]
#
#
# max_X1 = np.amax(X_train1, axis=0)
# if 0 in max_X1:
#     max_X1[max_X1 == 0] = 1
# X_train1 = np.divide(X_train1, max_X1)
#
# X_all = np.divide(X_all, max_X1)
#
# max_Y1 = np.max(Y_train1)/100
# if max_Y1 == 0:
#     max_Y1 = 1
# Y_train1 = np.divide(Y_train1, max_Y1)
#
# Y_all = np.divide(Y_all, max_Y1)
#
# x_all = torch.FloatTensor(X_all)
# y_all = torch.FloatTensor(Y_all)
# x_all = Variable(x_all)
# y_all = Variable(y_all)
#
# x1 = torch.FloatTensor(X_train1)
# y1 = torch.FloatTensor(Y_train1)
# x1 = Variable(x1)
# y1 = Variable(y1)
#
#
# net = Net(n_feature=n1, n_hidden=128, n_out=1)
# loss_func = torch.nn.MSELoss()
# optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.01)
#
# epochs = 300
# for i in range(epochs):
#     prediction = net(x1)
#     loss = loss_func(prediction, y1)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     print(i+1, "loss:", loss.data.item())
#
# testing_index1 = np.setdiff1d(np.array(range(N1)), training_index1)
# testing_data1 = whole_data1[testing_index1, :]
# X_test1 = testing_data1[:, 0:n1]
# X_test1 = np.divide(X_test1, max_X1)
# Y_test1 = testing_data1[:, n1][:, np.newaxis]
# Y_test1 = np.divide(Y_test1, max_Y1)
#
# x_test1 = torch.FloatTensor(X_test1)
# y_test1 = torch.FloatTensor(Y_test1)
# x_test1 = Variable(x_test1)
# y_test1 = Variable(y_test1)
#
#
# predict = net(x1)
# loss = loss_func(predict, y1)
# rel_error = np.mean(np.abs(np.divide(y1.detach().numpy().ravel() - predict.detach().numpy().ravel(), y1.detach().numpy().ravel())))
# print(rel_error * 100, end=' / ')
#
# predict1 = net(x_test1)
# loss1 = loss_func(predict1, y_test1)
# rel_error1 = np.mean(np.abs(np.divide(y_test1.detach().numpy().ravel() - predict1.detach().numpy().ravel(), y_test1.detach().numpy().ravel())))
# print(rel_error1 * 100, end=' / ')
#
# predict = net(x_all)
# loss = loss_func(predict, y_all)
# rel_error = np.mean(np.abs(np.divide(y_all.detach().numpy().ravel() - predict.detach().numpy().ravel(), y_all.detach().numpy().ravel())))
# print(rel_error * 100)


if __name__ == '__main__':
    result_dir = '/Users/weishouxin/Documents/Program/Python/database-transfer/20200906_mysql_results.csv'
    source_dir = '/Users/weishouxin/Documents/Program/Python/database-transfer/data/mysql-0905/mysql_5.5.csv'
    target_dir = '/Users/weishouxin/Documents/Program/Python/database-transfer/data/mysql-0905/mysql_8.0.csv'
    each_num = 10
    test_num = 94

    with open(result_dir, 'a') as f:
        f.write('DNN')
    target_file = pd.read_csv(target_dir)
    target_data = target_file.values

    target_X = target_data[:, :-1]
    target_Y = target_data[:, -1][:, np.newaxis]

    # min_x = np.min(target_X, axis=0)
    # max_x = np.max(target_X, axis=0)
    # _range = max_x - min_x
    # if 0 in _range:
    #     _range[_range == 0] = 1
    # target_X = np.divide(target_X - min_x, _range)
    #
    # min_y = np.min(target_Y, axis=0)
    # max_y = np.max(target_Y, axis=0)
    # _rangey = max_y - min_y
    # if 0 in _range:
    #     _rangey[_rangey == 0] = 1
    # target_Y = np.divide(target_Y - min_y, _rangey) * 100

    max_X1 = np.amax(target_X, axis=0)
    if 0 in max_X1:
        max_X1[max_X1 == 0] = 1
    target_X = np.divide(target_X, max_X1)


    max_Y1 = np.max(target_Y)/100
    if max_Y1 == 0:
        max_Y1 = 1
    target_Y = np.divide(target_Y, max_Y1)


    for N in range(1, 21):
        num_all = len(target_data)
        num_test = test_num
        num_train = each_num * N

        all_train_X = target_X[:-num_test, :]
        all_train_y = target_Y[:-num_test, :]
        test_data = target_X[-num_test:, :]
        test_label = target_Y[-num_test:, :]

        rel_error_list = []
        for i in range(5):
            time1 = time.time()

            permutation = np.random.permutation(num_all - num_test)
            train_index = permutation[0:num_train]
            train_data = all_train_X[train_index, :]
            train_label = all_train_y[train_index, :]

            x1 = torch.FloatTensor(train_data)
            y1 = torch.FloatTensor(train_label)
            x1 = Variable(x1)
            y1 = Variable(y1)

            m1, n1 = train_data.shape
            net = Net(n_feature=n1, n_hidden=128, n_out=1)
            loss_func = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.01)

            epochs = 300
            for i in range(epochs):
                prediction = net(x1)
                loss = loss_func(prediction, y1)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(i+1, "loss:", loss.data.item())

            x_test1 = torch.FloatTensor(test_data)
            y_test1 = torch.FloatTensor(test_label)
            x_test1 = Variable(x_test1)
            y_test1 = Variable(y_test1)

            predict = net(x_test1)
            # loss = loss_func(predict, y_test1)
            rel_error = np.mean(np.abs(np.divide(y_test1.detach().numpy().ravel() - predict.detach().numpy().ravel(), y_test1.detach().numpy().ravel())))

            rel_error_list.append(rel_error * 100)

            time2 = time.time()
            print((time2 - time1) * 1000)

        ave_rel_e = np.mean(rel_error_list)

        ave_rel_e = round(ave_rel_e, 3)
        with open('log.txt', 'a') as f:
            f.write('N = {}: '.format(N) + str(rel_error_list) + '  ' + str(ave_rel_e) + '\n')
        with open(result_dir, 'a') as f:
            f.write(', ' + str(ave_rel_e))
    with open(result_dir, 'a') as f:
        f.write('\n')
