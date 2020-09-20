import torch
import torch.nn as nn
import numpy as np
# import tensorflow as tf


class MMD_loss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        # print(type(source), type(target))
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            with torch.no_grad():
                XX = torch.mean(kernels[:batch_size, :batch_size])
                YY = torch.mean(kernels[batch_size:, batch_size:])
                XY = torch.mean(kernels[:batch_size, batch_size:])
                YX = torch.mean(kernels[batch_size:, :batch_size])
                loss = torch.mean(XX + YY - XY - YX)
            torch.cuda.empty_cache()
            return loss

if __name__ == '__main__':
    dir_data1 = '/Users/weishouxin/Documents/Program/Python/database-transfer/data/mysqlResult_transfer1.csv'
    print('Dataset: ' + dir_data1)
    whole_data1 = np.genfromtxt(dir_data1, delimiter=',', skip_header=1)
    (N1, n1) = whole_data1.shape
    n1 = n1 - 1
    X_all1 = whole_data1[:, 0:n1]
    X_all1 = torch.from_numpy(X_all1)

    dir_data2 = '/Users/weishouxin/Documents/Program/Python/database-transfer/data/mysql_AllNumeric.csv'
    print('Dataset: ' + dir_data2)
    whole_data2 = np.genfromtxt(dir_data2, delimiter=',', skip_header=1)
    (N2, n2) = whole_data2.shape
    n2 = n2 - 1
    X_all2 = whole_data2[:, 0:n2]
    X_all2 = torch.from_numpy(X_all2)
    mmd_loss = MMD_loss()
    loss = mmd_loss(X_all1, X_all2)
    print(loss)
