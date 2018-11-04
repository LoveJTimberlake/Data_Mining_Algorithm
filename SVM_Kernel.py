#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 15:18:19 2018

@author: justintimberlake
"""

import numpy as np

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
    
    
    
    
    
    
    