#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 15:18:19 2018

@author: justintimberlake
"""

import numpy as np
'''
class SVM:
    def __init__(self):
        pass
    
    def GetTrainData(self,data_x,data_y):
        pass 
    
    def Train(self,data_x,data_y):
        pass 
    
    def Predict(self,data_x):
        pass
    
    def SMO(self):
        pass 
'''   

def SimpleSMO(data,labels,C,toler,maxIter):    #toler是容错率 maxIter是最大迭代次数
    DataMat = np.mat(data)
    labelMat = np.at(labels).T
    b = 0 
    m,n = np.shape(DataMat)
    alphas = np.mat(np.zeros(m,1))
    iter = 0 
    while (iter < maxIter):
        alphaPairsChanged = 0 
        for i in range(m):
            fXi = np.float(np.multiply(alphas,labelMat).T * (DataMat * DataMat[i,:].T)) + b #multiply表示对应位置的元素相乘，fXi是分类的结果
            Ei = fXi - float(labelMat[i])
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i,m)
                fXj = float(np.multiply(alphas,labelMat).T *(DataMat * DataMat[j,:].T)) + b 
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy() 
                if(labelMat[i] != labelMat[j]):
                    L = max(0,alphas[j] - alphas[i])
                    H = min(C,C + alphas[j] - alphas[i])
                else:
                    L = max(0,alphas[j] + alphas[i] - C)
                    H = min(C,alphas[i] + alphas[j])
                if j == H :
                    print('L == H')
                    continue
                eta = 2.0 * DataMat[i,:] * DataMat[j,:].T - DataMat[i,:] * DataMat[i,:].T - DataMat[j,:] * DataMat[j,:].T
                if eta >= 0:
                    print('eta >= 0')
                    continue 
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta 
                alphas[j] = clipAlpha(alphas[j],H,L)
                if(np.abs(alphas[j] - alphaJold) < 0.00001):
                    print('j is not moving enough')
                    continue
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * DataMat[i,:] * DataMat[i,:].T - labelMat[j] * (alphas[j] - alphaJold) * DataMat[i,:] * DataMat[j,:].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * DataMat[i,:] * DataMat[j,:].T - labelMat[j] * (alphas[j] - alphaJold) * DataMat[j,:] * DataMat[j,:].T
                if(0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif(0 < alphas[j]) and (C>alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2)/2.0
                alphaPairsChanged += 1 
                print('iter: %d  i :%d ,pairs changed %d' % (iter,i,alphaPairsChanged))
        if(alphaPairsChanged == 0): 
            iter += 1
        else:
            iter = 0 
        print('iteration number:%d' %iter)
    return b,alphas

def loadDataSet(file):
    dataMat = [] 
    labelMat = []
    fr = open(file)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([np.float(lineArr[0]),np.float(lineArr[1])])
        labelMat.append(np.float(lineArr[2]))
    fr.close()
    return dataMat,labelMat
    
def selectJrand(i,m):
    j = i 
    while (j == i):
        j = int(np.random.uniform(0,m))
    return j 
    
def clipAlpha(aj,H,L):
    if aj > H:
        aj = H 
    if aj < L:
        aj = L 
    return aj 
    

#Platt SMO 
class optStruct:
    def __init__(self,DataMatIn,classlabels,C,toler):
        self.X = DataMatIn  #数据X
        self.labelMat = classlabels 
        self.C = C
        self.tol = toler 
        self.m = np.shape(DataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m,1)))     #数据实例个数的alpha
        self.b = 0 
        self.eCache = np.mat(np.zeros((self.m,2)))
    
def CalcEk(oS,k):   #oS是optStruct的结构体实例
    fxk = np.float(np.multiply(oS.alpha,oS.labelMat).T * (oS.X * oS.X[k,:].T)) + oS.b 
    Ek = fxk - np.float(oS.labelMat[k])
    return Ek
    
def selectJ(i,oS,Ei):
    maxK = -1 
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1,Ei]
    validEcacheList = np.nonzero(oS.eCache[:,0].A)[0]   #mat.A变成数组
    if(len(validEcacheList) > 1):
        for k in validEcacheList:
            if k == i:
                continue 
            Ek = CalcEk(oS,k)
            deltaE = abs(Ei - Ek)
            if(deltaE > maxDeltaE):
                maxK = k 
                maxDeltaE = deltaE 
                Ej = Ek 
        return maxK,Ej 
    else:
        j= selectJrand(i,oS.m)
        Ej = CalcEk(oS,j)
    return j,Ej
    
def updateEk(oS,k):
    Ek = CalcEk(oS,k)
    oS.eCache[k] = [1,Ek]

def innerL(i,oS):       #第二个alpha的循环(内循环)
    Ei = CalcEk(oS,i)
    if((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i,oS,Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if(oS.labelMat[i] != oS.labelMat[j]):
            L = max(0,oS.alphas[j] - oS.alphas[i])
            H = min(oS.C,oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0,oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print('L == H')
            return 0
        eta = 2.0 * oS.X[i,:] * oS.X[j,:].T - oS.X[i,:] * oS.X[i,:].T - oS.X[j,:] * oS.X[j,:].T
        if eta >= 0:
            print('eta >= 0') 
            return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS,j)
        if(abs(oS.alphas[j] - alphaJold) < 0.00001):
            print('J is moving enough')
            return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        updateEk(oS,j)
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i,:] * oS.X[i,:].T - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[i,:] * oS.X[j,:].T
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i,:] * oS.X[i,:].T - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[j,:] * oS.X[j,:].T
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1 
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2)/2.0
        return 1
    else:
        return 0
    




'''
部分解释：
关于if条件判断的理解：if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0))
首先，oS.tol是一个很小的小数，称作容错率，其存在是为了防止错误，我觉得类似于软间隔的e(松弛变量)
其次，完全可以将这个oS.tol换为0，我么就以换为0之后的条件来分析这个式子：
oS.labelMat[i] * Ei < 0 and (oS.alphas[i] < oS.C
拆开左边的式子：Yi*(fxi-Yi)=Yi*fxi-1<0, 从而有Yi*fxi<1。此时根据KKT条件，我们应该取alpha_i = C，但是右边显示alpha_i < C,所以违背了KKT条件
拆开右边的式子：Yi*(fxi-Yi)=Yi*fxi-1>0, 从而有Yi*fxi>1。此时根据KKT条件，我们应该取alpha_i = 0，但是右边显示alpha_i > C,所以违背了KKT条件
因此，此判断式是找出了违背了KKT条件的alpha
还有人问，为什么KKT条件有三个，此处只判断了两个？其实，此式确实判断了三个条件，只是合在了一起，下面是解释：
注意，oS.alphas[i] < oS.C包含了0<alpha_i<C和alpha_i=0两个条件（同理另一个也包含了alpha_i=C的情况），
所以alpha=0和alpha=C这两个KKT条件，被分别放在两个式子中判断了，0<alpha<C也被分成了两部分，这样三个条件就都有了判断

关于 L与H 
https://www.cnblogs.com/pinard/p/6111471.html 评论33楼
'''

def SmoP(DataMatIn, classLabels, C, toler, maxIter, kTup = ('lin',0)):
    oS = optStruct(np.mat(DataMatIn), np.mat(classLabels).transpose(),C,toler)
    iter_num = 0 
    entireSet = True 
    alphaPairsChanged = 0 
    while(iter_num < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0 
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i,oS)
                print("Fullset, iter_num: %s i:%s, pairs changed %s" %(iter_num,i,alphaPairsChanged))
            iter_num += 1 
        else:
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print("non_bound iter: %d  i:%d, pairs changed %d" %(iter_num,i,alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False
        elif(alphaPairsChanged == 0):
            entireSet = True
        print("iteration number: %d" %iter)
    return oS.b, oS.alphas

def clacWs(alphas,dataArr,classLabels):  #计算超平面参数w 
    X = np.mat(dataArr)
    labelMat = np.mat(classLabels).transpose()
    m,n = np.shape(X)
    w = np.zeros((n,1))
    for i in range(m):
        w += np.multiply(alphas[i] * labelMat[i], X[i,:].T)
    return w

def Classify_Data(dataArr,w_vector,b):
    dataMat = np.mat(dataArr)
    Result = [] 
    for i in range(np.shape(dataMat)[0]):
        r = (dataMat[i] * np.mat(w_vector) + b).A[0][0]
        if r > 0:
            r = 1 
        else:
            r = -1 
        Result.append(r)
    return Result


dataArr,labelArr = loadDataSet('train.txt')
b,alphas = SmoP(dataArr,labelArr,0.6,0.001,40)
w_vector = clacWs(alphas,dataArr,labelArr)
result = Classify_Data(dataArr,w_vector,b)

#添加核函数  radial basis function RBF的高斯版本

class New_optStruct:
    def __init__(self,DataMatIn,classlabels,C,toler,kTup):
        self.X = DataMatIn  #数据X
        self.labelMat = classlabels 
        self.C = C
        self.tol = toler 
        self.m = np.shape(DataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m,1)))     #数据实例个数的alpha
        self.b = 0 
        self.eCache = np.mat(np.zeros((self.m,2)))
        self.K = np.mat(np.zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i] = KernelTrans(self.X, self.X[i,:], kTup)

def KernelTrans(X,A,kTup):
    m,n = np.shape(X)
    K = np.mat(np.zeros((m,1)))
    if kTup[0] == 'lin':
        K = X * A.T 
    elif kTup[0] =='rbf':
        for j in range(m):
            deltaRow = X[j,:] - A 
            K[j] = deltaRow * deltaRow.T
        K = np.exp(K/(-1 * kTup[1] ** 2)) #kTup[1]是用户定义的到达率（函数值变为0的跌落速度)
    else:
        print("Kernel Name Error")
        return 0
    return K 

def innerL_Kernel(i,oS):       #第二个alpha的循环(内循环)
    Ei = CalcEk(oS,i)
    if((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i,oS,Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if(oS.labelMat[i] != oS.labelMat[j]):
            L = max(0,oS.alphas[j] - oS.alphas[i])
            H = min(oS.C,oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0,oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print('L == H')
            return 0
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j]
        if eta >= 0:
            print('eta >= 0') 
            return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS,j)
        if(abs(oS.alphas[j] - alphaJold) < 0.00001):
            print('J is moving enough')
            return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        updateEk(oS,j)
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i,j] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[i,j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i,j] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[i,j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1 
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2)/2.0
        return 1
    else:
        return 0

def CalcEk_Kernel(oS,k):   #oS是optStruct的结构体实例
    fxk = np.float(np.multiply(oS.alpha,oS.labelMat).T * oS.K[:,k] + oS.b)
    Ek = fxk - np.float(oS.labelMat[k])
    return Ek

def RBF_SVM(k1 = 1.3):      #k1(到达率)越小，需要的支持向量数越多 支持向量过多或过少都不好
    dataArr,labelArr = loadDataSet('train.txt')
    print('Input the k1 of RBF:(default value is 1.3)')
    k1 = input()
    b,alphas = smoP(dataArr,labelArr,200,0.0001,10000,('rbf',k1))
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas.A > 0)[0]     #支持向量索引
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd]
    print("there are %d support vectors" %(np.shape(sVs)[0]))
    m,n = np.shape(dataMat)
    errorCount = 0 
    for i in range(m):
        kernelEval = kernelTrans(sVs,dataMat[i,:],('rbf',k1))
        predict = kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b 
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1 
    print('The Train Error Rate is %d' %(errorCount * 1.0 / m))
    dataArr,labelArr = loadDataSet('test.txt')
    errorCount = 0
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    m,n = np.shape(dataMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,dataMat[i,:],('rbf',k1))
        predict = kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b 
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1 
    print('the test error rate is %d' %(errorCount * 1.0 / m))























    
    
    
    
    