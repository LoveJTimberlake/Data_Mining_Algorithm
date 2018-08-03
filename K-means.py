# coding=utf-8


import csv
import numpy as np
from numpy import genfromtxt	#将txt中内容变为字符串再变为指定类型   loadtxt只能够在一开始时选择一个格式且常为float
import matplotlib.pyplot as plt 
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


import scipy.spatial.distance as ssd

dict1 = {'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}

def Kmeans(dataset):
	def euclDistance(vector1,vector2):
		return sqrt(sum(pow(vector2 - vector1, 2)))

	def initCentroids(dataset,k):
		numSamples,dim = dataset.shape
		centroids = zeros((k,dim))
		for i in range(k):
			index = int(random.uniform(0,numSamples))
			centroids[i,:] = dataset[index,:]
		return centroids
	def kmeans(dataset,k):
		numSamples = dataset.shape[0]
		clusterAssment = mat(zeros((numSamples,2)))
		clusterChanged = True 

		centroids = initCentroids(dataset,k)

		while clusterChanged:
			clusterChanged = False
			for i in xrange(numSamples):
				minDist = 100000.0
				minIndex = 0
				for j in range(k):
					distance = euclDistance(centroids[j,:],dataset[i,:])
					if distance < minDist:
						minDist = distance 
						minIndex = j
				if clusterAssment[i,:] != minIndex:
					clusterChanged = True 
					clusterAssment[i,:] = minIndex, minDist **2

			for j in range(k):
				pointsInCluster = dataset[nonzero(clusterAssment[:,0].A == j)[0]]
				centroids[j,:] = mean(pointsInCluster,axis = 0)
		print('Complete')
		return centroids,clusterAssment


#K-means应用

dict = { '1':'assists_per_minuteReal', '2':' height','3':'time_playedReal','4' :'ageInteger','5':'points_per_minuteReal'}
from sklearn.cluster import KMeans 
from sklearn.cluster import Birch

Basketball_dataset = []
def load_data(file_path,X,Y):
	with open(file_path) as f:
		reader = csv.reader(f)
		for row in reader:
			Basketball_dataset.append([float(row[int(X)-1]),float(row[int(Y)-1])])
		f.close()
	print(Basketball_dataset)
if __name__ == '__main__':
	path = r'C:\Users\msi\mozart\basketball.csv'
	X = input("What data do you want to compare?\n1.Assist 2.height 3.time 4.age 5.point")
	Y = input("Second data:")
	load_data(path,X,Y)
	clf = KMeans(n_clusters = 3)
	y_pred = clf.fit_predict(Basketball_dataset)
	print(y_pred)
	x = [n[0] for n in Basketball_dataset]
	y = [n[1] for n in Basketball_dataset]
	plt.scatter(x,y,c = y_pred, marker = 'o')
	plt.xlabel(dict[X])
	plt.ylabel(dict[Y])
	plt.show()
