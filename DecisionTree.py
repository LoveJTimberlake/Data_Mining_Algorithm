#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 18:28:12 2018

@author: justintimberlake
"""

import numpy as np 
import math 
import operator
import pickle

#以下第一部分是基于ID3的决策树构建,无法直接处理数值型数据,而且没有加入剪枝步骤

def calcEntropy(dataSet):
    numEntries = len(dataSet)
    labelCount = {} 
    for featVec in dataSet:
        currentLabel = dataSet[-1]
        if currentLabel not in labelCount.keys():
            labelCount[currentLabel] = 0 
        labelCount[currentLabel] += 1 
    entrophy = 0
    for key in labelCount.keys():
        prob = float(labelCount[key])/numEntries
        entrophy -= prob * math.log(prob,2)
    return entrophy

def GiniFactor(dataSet):    #later to write
    pass

def LoadDataSet(file):
    dataMat = [] 
    labelMat = []
    fr = open(file)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([np.float(lineArr[0]),np.float(lineArr[1])])
        labelMat.append(np.float(lineArr[2]))
    fr.close()
    return dataMat,labelMat

def splitdataset(dataset,feat ,value):  #feat是整数，表示属性索引
    retDataSet = [] 
    for featvec in dataset:
        if featvec[feat] == value:
            reducedFeatVec = featvec[:feat]
            reducedFeatVec.extend(featvec[feat+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def choose_bestfeat_tosplit(dataSet):
    feat_num = np.shape(dataSet)[1] - 1 
    base_entrophy = calcEntropy(dataSet)
    bestIG = 0
    bestFeat = -1 
    for i in range(feat_num):
        feat_list = [x[i] for x in dataSet] #某个属性的所有值列表
        uniqueVals = set(feat_list)
        newEntrophy = 0
        for v in uniqueVals:
            subDataSet = splitdataset(dataSet,i,v)
            prob = len(subDataSet)*1.0 / len(dataSet)
            newEntrophy += prob * calcEntropy(subDataSet)
        ig = base_entrophy - newEntrophy
        if(ig > bestIG):
            bestIG = ig 
            bestFeat = i
    return bestFeat     #返回的是最佳属性的下标

def majorityCnt(classList):
    classCount = {} 
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0 
        classCount[vote] += 1
    soeredClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    return soeredClassCount[0][0]
    
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):     #即只有一个类别的样本集，则只返回一个结点（包含所有样本）即可
        return classList[0]
    if len(dataSet[0]) == 1:        #即只剩下一个属性但仍被调用创建子树的函数时，直接用该属性来划分
        return majorityCnt(classList)
    bestFeatIndex = choose_bestfeat_tosplit(dataSet)
    bestFeatLabel = labels[bestFeatIndex]
    myTree = {bestFeatLabel:{}}
    labels.remove(bestFeatLabel)
    featValues = [example[bestFeatIndex] for example in dataSet]
    uniquevalues = set(featValues)
    for value in uniquevalues:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitdataset(dataSet,bestFeatIndex,value),subLabels)  #嵌套的字典
    return myTree
    
#测试代码
datamat, labels = LoadDataSet('test.txt')
myTree = createTree(datamat,labels)
    
def classify(inputTree,featLabels,testvec):#testvec就是样本数据X
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testvec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key],featLabels,testvec)
            else:
                classLabel = secondDict[key]
    return classLabel
               
#在新任务中要能够使用之前就已经构造好的决策树 需要另外保存
def storeTree(inputTree,filename):
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    fr = open(filename)
    s = pickle.load(fr)
    fr.close()
    return s



#以下部分是CART回归树与模型树的构建，并会添加剪枝步骤
def loadDataSet(filename):
    dataMat = [] 
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine)
        dataMat.append(fltLine)
    return dataMat

def binSplitDataSet(dataSet,feat,value):    #将feat的value作为阈值然后进行数据集分割
    mat0 = dataSet[np.nonzero(dataSet[:,feat] > value)[0],:]    #nonzero(...)[0]是非零元素组成的数组 若没有时则为[]
    mat1 = dataSet[np.nonzero(dataSet[:,feat] <= value)[0],:]
    return mat0,mat1    
    
def regLeaf(dataSet):       #生成叶子模型，该叶子的判断值就是当前数据集上的均值
    return np.mean(dataSet[:,-1])

def regErr(dataSet):        #计算误差
    return np.var(dataSet[:,-1]) * np.shape(dataSet)[0]

def createCARTTree(dataSet, leafType = regLeaf, errType = regErr,ops = (1,4)):  #reg:回归 leaftype表示建立叶子结点的函数 errTypeb表示误差计算函数 ops则是包含建立树需要的参数元组
    feat,val = chooseBestSplit(dataSet,leafType,errType,ops)
    if feat == None:
        return val
    retTree = {} 
    retTree['spInd'] = feat
    retTree['spVal'] = val 
    lSet, rSet = binSplitDataSet(dataSet,feat,val)
    retTree['left'] = createCARTTree(lSet,leafType,errType,ops)
    retTree['right'] = createCARTTree(rSet,leafType,errType,ops)
    return retTree

def chooseBestSplit(dataSet,leafType = regLeaf, errType = regErr, ops = (1,4)):
    tolS = ops[0]
    tolN = ops[1]
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m,n = np.shape(dataSet)
    S = errType(dataSet)
    bestS = np.inf
    bestIndex = 0
    bestValue = 0 
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:,featIndex]):
            mat0,mat1 = binSplitDataSet(dataSet,featIndex,splitVal)
            if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
                continue 
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS 
    
    if S - bestS < tolS:
        return None, leafType(dataSet)
    mat0,mat1 = binSplitDataSet(dataSet,bestIndex,bestValue)
    if(np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):     #切分之后的数据集太少时则选择不切分
        return None, leafType(dataSet)
    return bestIndex,bestValue


#后剪枝
def isTree(obj):
    return (type(obj).__name__ == 'dict')

def getMean(tree):      #对当前子树树进行塌陷处理，每找到两个叶子结点就计算他们的平均值
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) * 1.0/ 2 

def Prune(tree,testData):       #testData是当前结点以下包含的样本集合
    if np.shape(testData)[0] == 0:  #没有测试数据
        return getMean(tree)
    if (isTree(tree['right'])) or (isTree(tree['left'])):
        lSet,rSet = binSplitDataSet(testData,tree['spInd'],tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = Prune(tree['left'],lSet)
    if isTree(tree['right']):
        tree['right'] = Prune(tree['right'],rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):      #当当前树的两个分支为叶子结点时,则考虑是否融合
        lSet , rSet = binSplitDataSet(testData,tree['spInd'], tree['spVal'])
        errorNoMerge = np.sum(np.power(lSet[:,-1] - tree['left'],2)) + np.sum(np.power(rSet[:-1] - tree['right'],2))    #此时的tree['left']与['right']都是Mean
        treeMean = (tree['left'] + tree['right']) * 1.0 / 2
        errorMerge = np.sum(np.power(testData[:-1] - treeMean),2)
        
        if errorMerge < errorNoMerge:
            print('merging')
            return treeMean
        else:
            return tree
    else:
        return tree

#测试代码
data = loadDataSet('train.txt')
data_mat = np.mat(data)
CARTTree= createCARTTree(data_mat)

test_data = loadDataSet('test.txt')
test_data_mat = np.mat(test_data)
final_CartTree = Prune(CARTTree,test_data_mat)

    
#模型树
def linearSolve(dataSet):
    m,n = np.shape(dataSet)
    X = np.mat(np.ones((m,n)))
    Y = np.mat(np.ones((m,1)))
    X[:,1:n] = dataSet[:,0:n-1]
    Y = dataSet[:,-1]
    xTx = X.T * X 
    if np.linalg.det(xTx) == 0.0:
        print("Error")
        return False
    ws = xTx.T * (X.T * Y)
    return ws,X,Y

def modelLeaf(dataSet):     #生成叶子结点 返回回归系数 直接替代regleaf()
    ws,X,Y = linearSolve(dataSet)
    return ws 

def modelErr(dataSet):      #被choosebestsplit()函数调用来寻找最佳的切分 直接替代regerr()
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws 
    return np.sum(np.power(Y - yHat),2)


#进行预测
def regTreeEval(model,inDat):
    return np.float(model)

def modelTreeEval(model,inDat):
    n = np.shape(inDat)[1]
    X = np.mat(np.ones((1,n+1)))
    X[:,1:n+1] = inDat
    return np.float(X * model)

def treeForeCast(tree,inData,modelEval = regTreeEval):
    if not isTree(tree):
        return modelEval(tree,inData)   
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'],inData,modelEval)
        else:
            return modelEval(tree['left'],inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'],inData,modelEval)
        else:
            return modelEval(tree['right'],inData)
        
def createForeCast(tree,testData,modelEval = regTreeEval):
    m = len(testData)
    yHat = np.mat(np.zeros((m,1)))
    for i in range(m):
        yHat[i,0] = treeForeCast(tree,np.mat(testData[i]),modelEval)
    return yHat
            
    

#创建回归树
train = np.mat(loadDataSet('train.txt'))
test = np.mat(loadDataSet('test.txt'))
regtree = createCARTTree(train,ops = (1,20))
yHat = createForeCast(regtree,test[:,0])
print(np.corrcoef(yHat,test[:,1],rowvar = 0)[0,1])

#创建模型树
modeltree = createCARTTree(train,modelLeaf,modelErr,(1,20))
yHat = createForeCast(modeltree,test[:,0],modelTreeEval)
print(np.corrcoef(yHat,test[:,1],rowvar = 0)[0,1])

#其结果越接近1，则效果越好    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
               
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
