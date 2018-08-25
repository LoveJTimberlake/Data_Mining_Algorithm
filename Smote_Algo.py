# coding=utf-8 

import random 
from sklearn.neighbors import NearestNeighbors 
import numpy as np 

class Smote:
	def __inti__(self,samples,N= 10, k = 5):
		self.n_samples, self.n_attrs = samples.shape 
		self.N = N 
		self.k = k 
		self.samples = samples 
		self.newindex = 0 		#counter of synthetic samples generated 

	def over_sampling(self):
		N = int(self.N/100)		#number of synthetic samples would be generated
		self.synthetic = np.zeros((self.n_samples * N, self.n_attrs))
		neighbors = NearestNeighbors(n_neighbors = self.k).fit(self.samples)
		#print('neighbors',neighbors)
		
		for i in range(len(self.samples)):
			nnarray = neighbors.kneighbors(self.samples[i].reshape(1,-1),return_distance = False)[0]	#算出其最近邻样本索引的数组并根据此构建一个新的样本
			#reshape(1,-1)是将不知道个数的返回的k个近邻样本数组变为一个一行k列的数组。（-1用于不知道原个数时)
			self.populate(N,i,nnarray)

		return self.synthetic

	def populate(self,N,i,nnarray):
		for j in range(N):
			nn = random.randint(0,self.k-1)		#a number used to choose a nearset neighbors of it 
			dif = self.samples[nnarray[nn]] - self.samples[i]		#nnarray[nn]表示一个近邻样本的索引
			factor = random.random()		# factor > 0 && factor < 1
			self.synthetic[self.newindex] = self.samples[i] + factor * dif 
			self.newindex += 1

def Test():
	a = np.array([[1,2,3],[4,5,6],[2,3,1],[2,1,2],[2,3,4],[2,3,4]])
	s = Smote(a,N = 100)
	s.over_sampling()



