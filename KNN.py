#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 13:53:12 2018

@author: justintimberlake
"""
import numpy as np 


class KNN_Classifiers:
    def __init__(self,k_neighbors = 5):
        self.TrainData_X = None 
        self.TrainData_Y = None #np.array形式
        self.k_neighbors = k_neighbors  #通过创建类实例时 a = KNN_Classifiers(10)来指定
        self.Predict_Y_Result = None 
        self.Mean_Vector = None 
        self.TrainData_CovMatrix = None 
    
    def Train(self,data_x,data_y):  #仅将训练数据(x,y)存入内存中 即可
        
        #计算训练集的均值向量
        self.Mean_Vector  = np.zeros(len(data_x[0]))
        for i in range(len(data_x)):
            self.Mean_Vector += np.array(data_x[i])
        self.Mean_Vector /= len(data_x)
    
        #计算训练集的协方差矩阵
        self.TrainData_X = np.array([x for x in data_x])
        self.TrainData_Y = np.array([y for y in data_y])
        
        Train_X = self.TrainData_X.T
        self.TrainData_CovMatrix = np.mat(np.cov(Train_X))
        
    def distance(self,uk_data,train_data):   #计算两个向量之间的马氏距离
        temp_mat = np.mat(np.array(uk_data) - self.Mean_Vector)
        return np.sqrt(temp_mat * self.TrainData_CovMatrix * temp_mat.T).getA()[0][0]
    
    def Predict(self,data_x):   #对未知数据中各个实例计算其与训练数据中最近的k个点并用最频繁的类别赋予给它，
        #返回的是与测试数据一一对应的类别数组
        result = []
        for uk_data in data_x:
            train_index = 0 
            distance_dict = {}  # [train_index]:distance
            for train_data in self.TrainData_X:
                distance_dict[train_index] = 0 
                distance_dict[train_index] = self.distance(uk_data,train_data)
                train_index += 1
                
            #排序得到k个最接近的数据点
            k_neighbors_index_list = [ x[0] for x in sorted(distance_dict.items(),key = lambda x : x[1], reverse = True)[:self.k_neighbors]]
            k_neighbors_y_array = np.array([self.TrainData_Y[x] for x in k_neighbors_index_list])
            Most_Possible_Result = np.argmax(np.bincount(k_neighbors_y_array))
            result.append(Most_Possible_Result)
        
        return np.array(result)
        

    
    
    
    
    
    
    
    
    
    
    