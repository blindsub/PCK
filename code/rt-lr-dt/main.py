import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
import time

def get_rt_moeld(X, Y1):
    # 性能预测模型使用决策回归树模型
    modle = DecisionTreeRegressor()

    modle.fit(X, Y1)

    # 存储训练好的回归树模型
    with open('models/regressor_tree.pickle', 'wb') as fw:
        pickle.dump(modle, fw)

    with open('models/regressor_tree.pickle', 'rb') as fr:
        new_modle = pickle.load(fr)

    test_pred = new_modle.predict(X)

    rel_error = np.mean(np.abs(np.divide(Y1.ravel() - test_pred.ravel(), Y1.ravel())))
    # print('rt model train rel_error(%):', rel_error * 100)

    # plt.figure()
    # plt.scatter(range(len(X)), Y1, s=20, edgecolor="black", c="darkorange")
    # plt.plot(range(len(X)), test_pred, color="yellowgreen", linewidth=2)
    # plt.show()


def get_lr_model(Y1, Y2):
    # 转换模型使用线形回归模型
    modle = linear_model.LinearRegression()

    modle.fit(Y1, Y2)

    # 存储训练好的线形回归模型
    with open('models/linear_regression.pickle', 'wb') as fw:
        pickle.dump(modle, fw)

    with open('models/linear_regression.pickle', 'rb') as fr:
        new_modle = pickle.load(fr)

    # train_pred = modle.predict(train_data)
    test_pred = new_modle.predict(Y1)

    rel_error = np.mean(np.abs(np.divide(Y2.ravel() - test_pred.ravel(), Y2.ravel())))
    # print('lr model train rel_error(%):', rel_error * 100)

    # plt.figure()
    # plt.scatter(range(len(Y1)), Y2, s=20, edgecolor="black", c="darkorange")
    # plt.plot(range(len(Y1)), test_pred, color="yellowgreen", linewidth=2)
    # plt.show()


def transfer(X, Y2):
    # 读取存储的回归树模型
    with open('models/regressor_tree.pickle', 'rb') as fr:
        rt_modle = pickle.load(fr)

    # 读取存储的线形回归模型
    with open('models/linear_regression.pickle', 'rb') as fr:
        lr_modle = pickle.load(fr)

    # 性能预测模型（回归树）预测在源域的结果
    y_predict_sour = rt_modle.predict(X).reshape(-1, 1)
    # 转换模型（线形回归）预测在目标域的结果
    y_predict_dist = lr_modle.predict(y_predict_sour)

    rel_error = np.mean(np.abs(np.divide(Y2.ravel() - y_predict_dist.ravel(), Y2.ravel())))
    # print('test model train rel_error(%):', rel_error * 100)

    return rel_error * 100

    # plt.figure()
    # a = plt.scatter(range(len(X)), Y2, marker='x')
    # b = plt.scatter(range(len(X)), y_predict_dist, marker='+')
    # plt.legend(handles=[a, b], labels=['real_y', 'pred_y'], loc='best')
    # plt.show()

if __name__ == '__main__':
    # num_all = 294
    # num_train = 50
    # num_test = 100

    result_dir = '/Users/weishouxin/Documents/Program/Python/database-transfer/20200906_mysql_results.csv'
    source_dir = '/Users/weishouxin/Documents/Program/Python/database-transfer/data/mysql-0905/mysql_5.5.csv'
    target_dir = '/Users/weishouxin/Documents/Program/Python/database-transfer/data/mysql-0905/mysql_8.0.csv'
    each_num = 10
    test_num = 94

    file = pd.read_csv(source_dir)
    data = file.values
    X = data[:, :-2]
    source_Y = data[:, -1][:, np.newaxis]

    file1 = pd.read_csv(target_dir)
    data1 = file1.values
    target_Y = data1[:, -1][:, np.newaxis]

    max_X1 = np.amax(X, axis=0)
    if 0 in max_X1:
        max_X1[max_X1 == 0] = 1
    source_X = np.divide(X, max_X1)

    max_Y1 = np.max(source_Y)
    if max_Y1 == 0:
        max_Y1 = 1
    source_Y = np.divide(source_Y, max_Y1)

    max_Y2 = np.max(target_Y)
    if max_Y2 == 0:
        max_Y2 = 1
    target_Y = np.divide(target_Y, max_Y2)

    # train_data, test_data, train_source_label, test_source_label, train_target_label, test_target_label = \
    #     train_test_split(X, source_Y, target_Y, train_size=num_train/num_all, test_size=num_test/num_all)

    # # 获取决策树模型
    # get_rt_moeld(train_data, train_source_label)
    #
    # # 获取线性回归模型
    # get_lr_model(train_source_label, train_target_label)
    #
    # # 迁移测试
    # transfer(test_data, test_target_label)

    with open(result_dir, 'a') as f:
        f.write('RT-LR')

    for N in range(1, 21):
        num_all = len(data)
        num_test = test_num
        num_train = each_num * N

        all_train_X = source_X[:-num_test, :]
        all_train_y1 = source_Y[:-num_test, :]
        all_train_y2 = target_Y[:-num_test, :]

        all_test_X = source_X[-num_test:, :]
        all_test_y1 = source_Y[-num_test:, :]
        all_test_y2 = target_Y[-num_test:, :]

        rel_error_list = []
        for i in range(5):
            time1 = time.time()

            permutation = np.random.permutation(num_all - num_test)
            train_index = permutation[0:num_train]
            train_data = all_train_X[train_index, :]
            train_label1 = all_train_y1[train_index, :]
            train_label2 = all_train_y2[train_index, :]

            get_rt_moeld(train_data, train_label1)
            get_lr_model(train_label1, train_label2)

            rel_error_list.append(transfer(all_test_X, all_test_y2))

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