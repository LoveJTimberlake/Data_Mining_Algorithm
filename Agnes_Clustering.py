# coding=utf-8

import csv
import numpy as np 
from numpy import *
from numpy import genfromtxt	#将txt中内容变为字符串再变为指定类型   loadtxt只能够在一开始时选择一个格式且常为float
import matplotlib.pyplot as plt 
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import matplotlib.pyplot as plt 
import seaborn as sns

def dist_func(dataset,cluster1,cluster2,Dist_Cal_Way):  #cluter是一个多维numpy数组
    if Dist_Cal_Way == 'dmin':
        dmin = 1e100
        for x in cluster1:
            for z in cluster2:
                dmin = min(dmin,np.sqrt(sum(pow(x - z, 2))))
        return dmin 

    elif Dist_Cal_Way == 'dmax':
        dmax = -1
        for x in cluster1:
            for z in cluster2:
                dmax = max(dmax,np.sqrt(sum(pow(x - z, 2))))
        return dmax 

    elif Dist_Cal_Way == 'davg':
        num_c1 = len(cluster1)
        num_c2 = len(cluster2)
        total_dist = 0
        for x in cluster1:
            for z in cluster2:
                total_dist += np.sqrt(sum(pow(x - z, 2)))
        return total_dist/(num_c1 * num_c2) 

    else :
        raise Exception("Wrong Parameter context for Dist_Cal_Way which is to set the way to calculate the distance between 2 clusters")


def Agnes(dataset,k,Dist_Cal_Way):
    
    #Use a dict to store cluster

    Cluster_Dict = dict()       #Cluster_Dict = {1:[]}

    for j in range(dataset.shape[0]):
        Cluster_Dict[j] = list()
        Cluster_Dict[j].append(dataset[j])         
    print("Init the cluster_dict successfully")
    M = np.mat(np.zeros((dataset.shape[0],dataset.shape[0])))

    for i in range(dataset.shape[0]):
        for j in range(i+1,dataset.shape[0]):
            M[i,j] = dist_func(dataset,Cluster_Dict[i],Cluster_Dict[j],Dist_Cal_Way)
            M[j,i] = M[i,j]
    print("Calculate the Dist Matrix first time")
    Current_Cluster_Num = dataset.shape[0]

    while(Current_Cluster_Num > k):
        print(Current_Cluster_Num)
        #Find two closest clusters
        closet_cluster1 = 0
        closet_cluster2 = 0
        Min_Dist = 1e100
        for i in Cluster_Dict.keys():
            for j in Cluster_Dict.keys():
                if(i==j):
                    continue
                current_dist = dist_func(dataset,Cluster_Dict[i],Cluster_Dict[j],Dist_Cal_Way)
                if(current_dist < Min_Dist):
                    closet_cluster1 = i
                    closet_cluster2 = j 
                    Min_Dist = current_dist
        
        #Merge cluster i and cluster j into cluster i
        #Then delete the cluster j in dict 
        for vec_j in Cluster_Dict[closet_cluster2]:
            Cluster_Dict[closet_cluster1].append(vec_j)
        
        del Cluster_Dict[closet_cluster2]

        for j in range(Current_Cluster_Num):
            if j not in Cluster_Dict.keys():
                continue
            else:
                M[closet_cluster1,j] = dist_func(dataset,Cluster_Dict[closet_cluster1],Cluster_Dict[j],Dist_Cal_Way)
                M[j,closet_cluster1] = M[closet_cluster1,j]
        
        Current_Cluster_Num -= 1
    
    return Cluster_Dict

Iris_Data = load_iris()['data']
Cluster_Dict = Agnes(Iris_Data,3,'davg')
print(Cluster_Dict)
print(Cluster_Dict.keys())
samples_num = 0
for k in Cluster_Dict.keys():
    samples_num += len(Cluster_Dict[k])
print(samples_num)











