# coding=utf-8

import numpy as np 

def loadDataSet(file):
	fr = open(file)
	dataMat = [] 
	labelMat = []
	one_vector = fr.readline().split('\t')
	feat_num = len(one_vector) - 1 
	for line in fr.readlines():
		lineArr = [] 
		curArr = line.strip().split('\t')
		for i in range(feat_num):
			lineArr.append(np.float(curArr[i]))
		dataMat.append(lineArr)
		labelMat.append(np.float(curArr[-1]))
	return dataMat,labelMat

#标准回归
def standRegres(xArr,yArr):
	xMat = np.mat(xArr)
	yMat = np.mat(yArr).T
	xTx = xMat.T * xMat
	if np.linalg.det(xTx) == 0.0:
		print("Singular Matrix")
		return 
	ws = xTx.T * (xMat.T * yMat)
	return ws 

#test main
xArr, yArr = loadDataSet('train.txt')
ws = standRegres(xArr, yArr)
xMat = np.mat(xArr)
yMat = np.mat(yArr)
yHat = xMat * ws 


#局部加权线性回归
def lwlr(testPoint,xArr,yArr,k = 1.0):
	xMat = np.mat(xArr)
	yMat = np.mat(yArr)
	m = xMat.shape[0]
	weights = np.mat(np.eye((m)))  #对角矩阵
	for j in range(m):
		diffMat = testPoint - xMat[j,:]
		weights[j,j] = np.exp(diffMat * diffMat.T/(-2.0 * k ** 2))

	xTx = xMat.T * (weights * xMat)
	if np.linalg.det(xTx) == 0.0:
		print("Singular Matrix")
		return 

	ws = xTx.I * (xMat.T * (weights * yMat))

	return testPoint * ws 

def lwlrTest(testArr,xArr,yArr,k = 1.0 ):
	m = testArr.shape[0]
	yHat = np.zeros(m)
	for i in range(m):
		yHat[i] = lwlr(testArr[i],xArr,yArr,k)
	return yHat

xArr, yArr = loadDataSet('train.txt')
yHat = lwlrTest(xArr,xArr,yArr,0.003)



#Ridge Regreesion  特征数比样本数多

def ridgeRegres(xMat,yMat,lam = 0.2):
	xTx = xMat.T * xMat
	denom = xTx + np.eye(xMat.shape[1]) * lam 

	if np.linalg.det(denom) == 0.0:
		print("Singular Matrix")
		return 

	ws = denom.T * (xMat.T * yMat)
	return ws 

def ridgeTest(xArr,yArr):
	xMat = np.mat(xArr)
	yMean = np.mean(yMat,0)
	yMat = yMat - yMean
	xMeans = np.mean(xMat,0)
	xVar = np.var(xMat,0)
	xMat = (xMat - xMeans)/xVar
	numTestPts = 30 
	wMat = np.zeros((numTestPts,xMat.shape[1]))
	for i in range(numTestPts):
		ws = ridgeRegres(xMat,yMat,exp(i-10))
		wMat[i,:] = ws.T 
	return wMat





















