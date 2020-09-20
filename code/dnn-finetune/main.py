import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.optim
import mmd
from modle import Net
import pandas as pd
import time


# print('Read whole dataset from csv file ...')
# dir_data = 'data/mysqlResult_5.5.csv'
# print('Dataset: ' + dir_data)
# whole_data = np.genfromtxt(dir_data, delimiter=',', skip_header=1)
# (N, n) = whole_data.shape
# n = n - 1
#
# N_train = N - 200
# permutation = np.random.permutation(N)
# training_index = permutation[0:N_train]
# training_data = whole_data[training_index, :]
# X_train = training_data[:, 0:n]
# Y_train = training_data[:, n][:, np.newaxis]
#
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
# # X_train1 = whole_data1[:, 0:n1]
# # Y_train1 = whole_data1[:, n1][:, np.newaxis]
#
# X_all = whole_data1[:, :n]
# Y_all = whole_data1[:, n][:, np.newaxis]
#
#
# max_X = np.amax(X_train, axis=0)
# if 0 in max_X:
#     max_X[max_X == 0] = 1
# X_train = np.divide(X_train, max_X)
#
# max_X1 = np.amax(X_train1, axis=0)
# if 0 in max_X1:
#     max_X1[max_X1 == 0] = 1
# X_train1 = np.divide(X_train1, max_X1)
#
# # max_X_all = np.amax(X_all, axis=0)
# # if 0 in max_X_all:
# #     max_X_all[max_X_all == 0] = 1
# X_all = np.divide(X_all, max_X1)
#
# max_Y = np.max(Y_train)/100
# if max_Y == 0:
#     max_Y = 1
# Y_train = np.divide(Y_train, max_Y)
#
# max_Y1 = np.max(Y_train1)/100
# if max_Y1 == 0:
#     max_Y1 = 1
# Y_train1 = np.divide(Y_train1, max_Y1)
#
# # max_Y_all = np.max(Y_all)/100
# # if max_Y_all == 0:
# #     max_Y_all = 1
# Y_all = np.divide(Y_all, max_Y1)
#
# X_train0 = X_train[:, :]
# Y_train0 = Y_train[:, :]
#
# x_all = torch.FloatTensor(X_all)
# y_all = torch.FloatTensor(Y_all)
# x_all = Variable(x_all)
# y_all = Variable(y_all)
#
# x = torch.FloatTensor(X_train0)
# y = torch.FloatTensor(Y_train0)
# x = Variable(x)
# y = Variable(y)
#
# x1 = torch.FloatTensor(X_train1)
# y1 = torch.FloatTensor(Y_train1)
# x1 = Variable(x1)
# y1 = Variable(y1)
#
#
# net = Net(n_feature=n, n_hidden=128, n_out=1)
# # print(net)
# net.load_state_dict(torch.load("models/my_model.pkl"))
# loss_func = torch.nn.MSELoss()
# optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.01)
# # mmd_loss = mmd.MMD_loss()
#
# epochs = 300
# for i in range(epochs):
#     prediction = net(x1)
#     # print(prediction)
#     # loss = loss_func(prediction, y) + mmd_loss(net(x_all)[1], net(x1)[1])
#     loss = loss_func(prediction, y1)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     print(i+1, "loss:", loss.data.item())
#
# # torch.save(net, "my_model.pkl")
# # # net = torch.load("my_model.pkl")
# # testing_index = np.setdiff1d(np.array(range(N)), training_index)
# # testing_data = whole_data[testing_index, :]
# # X_test = testing_data[:, 0:n]
# # X_test = np.divide(X_test, max_X)
# # Y_test = testing_data[:, n][:, np.newaxis]
# # Y_test = np.divide(Y_test, max_Y)
# #
# # x_test = torch.FloatTensor(X_test)
# # y_test = torch.FloatTensor(Y_test)
# # x_test = Variable(x_test)
# # y_test = Variable(y_test)
# #
# # predict = net(x)
# # loss = loss_func(predict, y)
# # rel_error = np.mean(np.abs(np.divide(y.detach().numpy().ravel() - predict.detach().numpy().ravel(), y.detach().numpy().ravel())))
# # # print(loss.data.item())
# # print(rel_error * 100, end=' / ')
#
# # predict = net(x_test)
# # loss = loss_func(predict, y_test)
# # # print()
# # rel_error = np.mean(np.abs(np.divide(y_test.detach().numpy().ravel() - predict.detach().numpy().ravel(), y_test.detach().numpy().ravel())))
# # # print(loss.data.item())
# # print(rel_error * 100, end=' / ')
#
# testing_index1 = np.setdiff1d(np.array(range(N1)), training_index1)
# testing_data1 = whole_data1[testing_index1, :]
# X_test1 = testing_data1[:, 0:n]
# X_test1 = np.divide(X_test1, max_X1)
# Y_test1 = testing_data1[:, n][:, np.newaxis]
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
# # print()
# rel_error = np.mean(np.abs(np.divide(y1.detach().numpy().ravel() - predict.detach().numpy().ravel(), y1.detach().numpy().ravel())))
# # print(loss.data.item())
# print(rel_error * 100, end=' / ')
# # with open('log.txt', 'a') as f:
# #     f.write(str(rel_error * 100) + ' / ')
#
# predict1 = net(x_test1)
# loss1 = loss_func(predict1, y_test1)
# rel_error1 = np.mean(np.abs(np.divide(y_test1.detach().numpy().ravel() - predict1.detach().numpy().ravel(), y_test1.detach().numpy().ravel())))
# # print()
# # print(loss1.data.item())
# print(rel_error1 * 100, end=' / ')
# # with open('log.txt', 'a') as f:
# #     f.write(str(rel_error1 * 100) + ' / ')
#
# predict = net(x_all)
# loss = loss_func(predict, y_all)
# # print()
# rel_error = np.mean(np.abs(np.divide(y_all.detach().numpy().ravel() - predict.detach().numpy().ravel(), y_all.detach().numpy().ravel())))
# # print(loss.data.item())
# print(rel_error * 100)
# # with open('log.txt', 'a') as f:
# #     f.write(str(rel_error * 100) + '\n')

if __name__ == '__main__':
    result_dir = '/Users/weishouxin/Documents/Program/Python/database-transfer/20200906_mysql_results.csv'
    source_dir = '/Users/weishouxin/Documents/Program/Python/database-transfer/data/mysql-0905/mysql_5.5.csv'
    target_dir = '/Users/weishouxin/Documents/Program/Python/database-transfer/data/mysql-0905/mysql_8.0.csv'
    each_num = 10
    test_num = 94

    with open(result_dir, 'a') as f:
        f.write('DNN-Finetune')
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
            net.load_state_dict(torch.load("models/my_model.pkl"))
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
            loss = loss_func(predict, y_test1)
            rel_error = np.mean(np.abs(np.divide(y_test1.detach().numpy().ravel() - predict.detach().numpy().ravel(), y_test1.detach().numpy().ravel())))

            rel_error_list.append(rel_error * 100)

            time2 = time.time()
            print((time2-time1)*1000)

        ave_rel_e = np.mean(rel_error_list)

        ave_rel_e = round(ave_rel_e, 3)
        with open('log.txt', 'a') as f:
            f.write('N = {}: '.format(N) + str(rel_error_list) + '  ' + str(ave_rel_e) + '\n')
        with open(result_dir, 'a') as f:
            f.write(', ' + str(ave_rel_e))
    with open(result_dir, 'a') as f:
        f.write('\n')