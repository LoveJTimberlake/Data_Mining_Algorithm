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
import random
import queue

def dist_func(vec1,vec2):
    return np.sqrt(sum(pow(vec2 - vec1, 2)))


def Cal_Near_Area(dataset,x,n,r):
    
    Near_Sample_Set = list()

    for i in range(n):

        if (dataset[i] == x).all():
            continue

        if dist_func(x,dataset[i]) <= r:
            Near_Sample_Set.append(dataset[i])

    return Near_Sample_Set

def judge_vec_equal(vec1,vec2):
    for i in range(len(vec1)):
        if vec1[i] != vec2[i]:
            return False
    return True

def Judge_Vec_In_Vecs(x,dataset):
    for to_compare_vec in dataset:
        if judge_vec_equal(x,to_compare_vec):
            return True 
    return False


def DBSCAN(dataset,r,MinPts):
    Core_Object_Set = list()
    Cluster_Samples_Dict = dict()
    n = dataset.shape[0]

    for i in range(n):
        near_area = Cal_Near_Area(dataset,dataset[i],n,r)
        if len(near_area) >= MinPts:
            Core_Object_Set.append(dataset[i])
    print("Choose all %d Core_Objects Already"%(len(Core_Object_Set)))
    k = 0
    Unadded_Samples_Set = dataset.copy()            #Array is not hashable so we need to check whether there is a dunplicate
    
    while len(Core_Object_Set):
        Old_Unadded_Samples_Set = Unadded_Samples_Set.copy()

        #Choose a core-object randomly and init the queue 
        inited_core_object = dataset[random.randrange(0,n)]
        q = queue.Queue()
        q.put(inited_core_object)
        print("Chose a inited_Core_Object while there are %d Core_Objects"%(len(Core_Object_Set)))
        #delete inited_core_object from Unadded_Samples_Set
        Unadded_Samples_Set = [v for v in Unadded_Samples_Set if judge_vec_equal(v,inited_core_object)]

        while not q.empty():
            first_sample_inqueue = q.get()
            Ns = Cal_Near_Area(dataset,first_sample_inqueue,n,r)
            print("Find the near_area for the first sample q in Q(the queue)")
            Delta = list()
            if len(Ns) >= MinPts:
                print("This is a trustable core_object")
                #   Delta = Ns 与 Unadded_Samples_Set 的交集
                for near_sample in Ns:
                    if Judge_Vec_In_Vecs(near_sample,Unadded_Samples_Set):      #Doubt to manage
                        Delta.append(near_sample)
                
                #Put samples in Delta in Q
                for chosen_sample_Delta in Delta:
                    q.put(chosen_sample_Delta)

                # Unadded_Samples_Set = Unadded_Samples_Set - Delta
                New_Unadded_Sampels_Set = [v for v in Unadded_Samples_Set if not Judge_Vec_In_Vecs(v,Delta)]
                Unadded_Samples_Set = New_Unadded_Sampels_Set
        
        k += 1
        Cluster_Samples_Dict[k] = [v for v in Old_Unadded_Samples_Set if not Judge_Vec_In_Vecs(v,Unadded_Samples_Set)]
        Core_Object_Set = [v for v in Core_Object_Set if not Judge_Vec_In_Vecs(v,Cluster_Samples_Dict[k])]
        print(Core_Object_Set)
    return Cluster_Samples_Dict

Iris_Data = load_iris()['data']
Cluster_Result= DBSCAN(Iris_Data,1,40)      #傻x参数 难调死了
print(Cluster_Result)



