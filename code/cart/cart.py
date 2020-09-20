import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def loadDataSet(filename):
    target_file = pd.read_csv(filename)
    target_data = target_file.values

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
    # data = np.hstack((target_X, target_Y))

    # dataMat = []
    # fr = open(filename)
    # for line in fr.readlines():
    #     curLine = line.strip().split('\t')
    #     fltLine = []
    #     for i in curLine:
    #         fltLine.append(float(i))
    #     dataMat.append(fltLine)

    return target_data.tolist()


def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1

def regLeaf(dataSet):
    return np.mean(dataSet[:, -1])

def regErr(dataSet):
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]

def chooseBestSplit(dataSet, leafType=regLeaf, errType =regErr, ops=(1, 4)):
    tolS = ops[0]
    tolN = ops[1]

    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)

    m, n = np.shape(dataSet)
    S = errType(dataSet)
    bestS = np.inf
    bestIndex = 0
    bestValue = 0

    for featIndex in range(n-1):
        for splitVal in set(dataSet[:, featIndex].T.A.tolist()[0]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestS = newS
                bestIndex = featIndex
                bestValue = splitVal

    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
        return None, leafType(dataSet)

    return bestIndex, bestValue

def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None:
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, regLeaf, regErr, ops)
    retTree['right'] = createTree(rSet, regLeaf, regErr, ops)

    return retTree

def isTree(obj):
    return (type(obj).__name__ == 'dict')

def getMean(tree):
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])

    return (tree['left'] + tree['right']) / 2.0

def prune(tree, testData):
    if np.shape(testData)[0] == 0:
        return getMean(tree)
    if isTree(tree['left']) or isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMege = sum(np.power(lSet[:, -1] - tree['left'], 2)) + sum(np.power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = sum(np.power(testData[:, -1] - treeMean, 2))
        if errorMerge < errorNoMege:
            return treeMean
        else:
            return tree
    else:
        return tree


def regTreeEval(model, inDat):
    return float(model)


def treeForeCast(tree, inData, modelEval=regTreeEval):
    # print(type(inData))
    # print(tree['spInd'])
    # print(inData[tree['spInd']])
    inData = np.array(inData)[0]

    if not isTree(tree):
        return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            inData = np.mat(inData)
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            inData = np.mat(inData)
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            inData = np.mat(inData)
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            inData = np.mat(inData)
            return modelEval(tree['right'], inData)

def createForeCast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = np.mat(np.zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, np.mat(testData[i]), modelEval)
    return yHat

if __name__ == '__main__':
    num_all = 454
    num_train = 200 + 50
    num_test = 200

    all_data = loadDataSet('data/mysqlResult_transfer2.csv')
    train_data, test_data = \
        train_test_split(all_data, train_size=num_train / num_all, test_size=num_test / num_all)


    trainMat = np.mat(train_data[:num_train-50])
    validMat = np.mat(train_data[-50:])
    testMat = np.mat(test_data)

    myTree = createTree(trainMat)
    # print(myTree)
    myTree = prune(myTree, validMat)
    # print(myTree)

    yHat = createForeCast(myTree, testMat[:, :-1])
    rel_error = np.mean(np.abs(np.divide(testMat[:, -1] - yHat, testMat[:, -1])))
    print('target test rel_error(%):', rel_error * 100)