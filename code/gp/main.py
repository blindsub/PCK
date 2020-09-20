import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import time


def get_source_model(X, Y):
    train_data = X
    train_label = Y

    # 核函数的取值
    # kernel = C(0.1, (0.001, 0.1)) * RBF(0.5, (1e-4, 10))
    # kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))

    # 创建高斯过程回归,并训练
    # model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1)
    model = GaussianProcessRegressor()

    # 训练
    model.fit(train_data, train_label)

    train_pred = model.predict(train_data)
    rel_error = np.mean(np.abs(np.divide(train_label.ravel() - train_pred.ravel(), train_label.ravel())))
    # print('source train rel_error(%):', rel_error * 100)

    joblib.dump(model, 'models/gaussian_process_regressor_source.pickle')

    # plt.figure()
    # plt.scatter(range(len(train_data)), train_label, s=20, edgecolor="black", c="darkorange")
    # plt.plot(range(len(train_data)), train_pred, color="yellowgreen", linewidth=2)
    # plt.title('source train label and train pred')
    # plt.show()


def get_target_model(X, Y):
    train_data = X
    train_label = Y
    model = joblib.load('models/gaussian_process_regressor_source.pickle')

    # 对源域模型用目标域数据进行训练
    model.fit(train_data, train_label)

    train_pred = model.predict(train_data)

    rel_error = np.mean(np.abs(np.divide(train_label.ravel() - train_pred.ravel(), train_label.ravel())))
    # print('target train rel_error(%):', rel_error * 100)

    joblib.dump(model, 'models/gaussian_process_regressor_target.pickle')

    # plt.figure()
    # plt.scatter(range(len(train_data)), train_label, s=20, edgecolor="black", c="darkorange")
    # plt.plot(range(len(train_data)), train_pred, color="yellowgreen", linewidth=2)
    # plt.title('target train label and train pred')
    # plt.show()


def test_target_model(X, Y):
    test_data = X
    test_label = Y

    # 读取训练好的目标域模型
    model = joblib.load('models/gaussian_process_regressor_target.pickle')

    test_pred = model.predict(test_data)

    rel_error = np.mean(np.abs(np.divide(test_label.ravel() - test_pred.ravel(), test_label.ravel())))
    # print('target test rel_error(%):', rel_error * 100)

    return rel_error * 100

    # plt.figure()
    # plt.scatter(range(len(test_data)), test_label, s=20, edgecolor="black", c="darkorange")
    # plt.plot(range(len(test_data)), test_pred, color="yellowgreen", linewidth=2)
    # plt.title('target test label and train pred')
    # plt.show()


if __name__ == '__main__':
    # num_all = 454
    # num_train = 50
    # num_test = 200

    result_dir = '/Users/weishouxin/Documents/Program/Python/database-transfer/20200906_mysql_results.csv'
    source_dir = '/Users/weishouxin/Documents/Program/Python/database-transfer/data/mysql-0905/mysql_5.5.csv'
    target_dir = '/Users/weishouxin/Documents/Program/Python/database-transfer/data/mysql-0905/mysql_8.0.csv'
    each_num = 10
    test_num = 94

    # 获得源域模型
    source_file = pd.read_csv(source_dir)
    source_data = source_file.values
    source_X = source_data[:, :-1]
    source_Y = source_data[:, -1][:, np.newaxis]

    # 数据预处理
    max_X1 = np.amax(source_X, axis=0)
    if 0 in max_X1:
        max_X1[max_X1 == 0] = 1
    source_X = np.divide(source_X, max_X1)

    max_Y1 = np.max(source_Y)
    if max_Y1 == 0:
        max_Y1 = 1
    source_Y = np.divide(source_Y, max_Y1)

    get_source_model(source_X, source_Y)

    target_file = pd.read_csv(target_dir)
    target_data = target_file.values
    target_X = target_data[:, :-1]
    target_Y = target_data[:, -1][:, np.newaxis]

    max_X2 = np.amax(target_X, axis=0)
    if 0 in max_X2:
        max_X2[max_X2 == 0] = 1
    target_X = np.divide(target_X, max_X2)

    max_Y2 = np.max(target_Y) / 100
    if max_Y2 == 0:
        max_Y2 = 1
    target_Y = np.divide(target_Y, max_Y2)

    # # 用目标域部分数据进行微调
    # target_file = pd.read_csv('data/mysqlResult_transfer2.csv')
    # target_data = target_file.values
    # target_X = target_data[:, :-1]
    # target_Y = target_data[:, -1][:, np.newaxis]
    #
    # # 数据预处理
    # max_X2 = np.amax(target_X, axis=0)
    # if 0 in max_X2:
    #     max_X2[max_X2 == 0] = 1
    # target_X = np.divide(target_X, max_X2)
    #
    # max_Y2 = np.max(target_Y)
    # if max_Y2 == 0:
    #     max_Y2 = 1
    # target_Y = np.divide(target_Y, max_Y2)
    #
    # train_data, test_data, train_label, test_label = \
    #     train_test_split(target_X, target_Y, train_size=num_train/num_all, test_size=num_test/num_all)
    # get_target_model(train_data, train_label)
    #
    # # 用目标域剩余数据进行测试
    # test_target_model(test_data, test_label)

    with open(result_dir, 'a') as f:
        f.write('GP')

    for N in range(1, 21):
        num_all = len(target_data)
        num_test = test_num
        num_train = each_num * N

        all_train_X = target_X[:-num_test, :]
        all_train_y = target_Y[:-num_test, :]
        all_test_X = target_X[-num_test:, :]
        all_test_y = target_Y[-num_test:, :]

        rel_error_list = []
        for i in range(5):
            time1 = time.time()

            permutation = np.random.permutation(num_all-num_test)
            train_index = permutation[0:num_train]
            train_data = all_train_X[train_index, :]
            train_label = all_train_y[train_index, :]

            get_target_model(train_data, train_label)

            rel_error_list.append(test_target_model(all_test_X, all_test_y))

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

