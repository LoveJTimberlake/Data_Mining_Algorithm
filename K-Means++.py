# coding=utf-8
import pandas as pd 
import numpy as np 
from numpy import *
import sklearn 
from sklearn import dataset 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from sklearn import metrics
import math 
import random
from sklearn.datasets import load_iris

def euclDistance(vector1,vector2):
    return np.sqrt(sum(pow(vector2 - vector1, 2)))

def InitCentroids_KMeansPP(dataset,k):

    #Select a data as the first random centroid
    n = dataset.shape[0]
    first_random_cent = dataset[random.randrange(0,n)]
    
    selected_cent_num = 1
    Chosen_Cents_List = list()
    Chosen_Cents_List.append(first_random_cent)
    while(selected_cent_num < k):
        Max_Dist = -1
        Next_Chosen_Cent = np.array([])
        for data in dataset:
            #Calculate the min dist between this data and all Chosen Centroids
            min_dist = 1e100
            for chosen_cent in Chosen_Cents_List:
                min_dist = min(min_dist,euclDistance(data,chosen_cent))
            #Choose data with the max(min_dist) to be next cent 
            if Max_Dist < min_dist:
                Max_Dist = min_dist
                Next_Chosen_Cent = data   
        Chosen_Cents_List.append(Next_Chosen_Cent)
        selected_cent_num += 1
    return np.array(Chosen_Cents_List)

def kmeans(dataset,k):
    numSamples = dataset.shape[0]
    clusterAssment = mat(zeros((numSamples,2)))
    clusterChanged = True 

    centroids = InitCentroids_KMeansPP(dataset,k)   #Here we choose new way to init centroids in K-Means++ to use in K-Means

    while clusterChanged:  #每次迭代后检查质心有否移动.
        clusterChanged = False
        for i in range(numSamples):
            minDist = 100000000000000000000000.0
            minIndex = 0
            for j in range(k):  #计算每个样本与哪个簇心最近. 并记录最近距离与最近质心标记.
                distance = euclDistance(centroids[j,:],dataset[i,:])
                if distance < minDist:
                    minDist = distance 
                    minIndex = j
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True 
                clusterAssment[i,:] = minIndex, minDist **2

        for j in range(k):  
            #重新划分并更新簇心.
            pointsInCluster = dataset[nonzero(clusterAssment[:,0].A == j)[0]]
            centroids[j,:] = mean(pointsInCluster,axis = 0)
            
            #画图
            x = pointsInCluster[:,0]
            y = pointsInCluster[:,1]
            c = 'g'
            if(j == 1):
                c = 'r'
            elif j == 2:
                c = 'b'
            plt.scatter(x,y,marker = 'o',color = c)
            
            #画质心
            plt.scatter(centroids[j,0],centroids[j,1],marker = 'x',color = c, linewidths=100)
            plt.xlabel('Feature1',fontsize=14)
            plt.ylabel('Feature2',fontsize=14)
        plt.show()


    return centroids,clusterAssment

def K_MeansPlusPlus(dataset,k):
    centroids,clusterAssement = kmeans(dataset,k)
    return centroids,clusterAssement

Iris_Data = load_iris()['data'][:,[0,1]]
centroids,clusterAssment = K_MeansPlusPlus(Iris_Data,3)
print(clusterAssment)

