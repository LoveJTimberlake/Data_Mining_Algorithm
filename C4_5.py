# coding=utf-8
import operator 
from numpy import *
from math import log

##########################################

#算法理解：
#使用增益率(= 增益/熵)来衡量条件测试的好坏,选择最大的
#一个分支下不可再分或全部实例为pure(该分支的条件下目标值全部一样)时则为叶节点，叶节点中占据最大比例的类为该分支的类别
#剪枝的准则：Pessimistic Pruning 通过递归计算目标节点分支的错误率来获得该目标节点的错误率，从而估算未知实例上的错误率


##########################################

class C45_DecisionTree:
	def __init__(self):
		self.Tree = {}
		self.AttributeList = []	

	def GetAttribute(self,file):	#获取属性个数
		fr = open(file,'r')
		attr_line = fr.readline()
		attr_list = (attr_line.strip()).split()
		return len(attr_list)
	
	def Init_DataSet(self,filename):		#规整化数据集		

		 #dataset中的数据以一行一个实例，一列一个属性值组织
		fr = open(filename,'r')
		Samples = fr.readlines()
		Samples_Num = len(Samples)
		file = input("Data file location:")
		attr_num = GetAttribute(file)
		S_Matrix = zeros((Samples_Num,attr_num))
		ClassLabelVector = []
		row = 0
		for Sample in Samples:
			Sample = Sample.strip()
			Sample_List = Sample.split()
			S_Matrix[row,:] = Sample_List[0:attr_num]
			ClassLabelVector.append(Sample_List[-1])
			row += 1

		DataSet= []
		row = 0
		for row in range(0,Samples_Num):
			temp = list(S_Matrix[row, : ])
			temp.append(ClassLabelVector[row])
			DataSet.append(temp)

		return DataSet

	def SplitDataSet(self,DataSet, Axis, Value):		#提取特定数据集(筛选第Axis个元素满足特定条件的元组并省略其值)，DataSet由Init_DataSet()获得
		Ret_DataSet = []
		for FeatVec in DataSet:
			if FeatVec[Axis] == Value:
				ReducedFeatVec = FeatVec[: Axis]
				ReducedFeatVec.extend(FeatVec[Axis+1:])
				Ret_DataSet.append(ReducedFeatVec)

		return Ret_DataSet

	def CalShannonEnt(self,DataSet):		#计算香农熵
		Samples_Num = len(DataSet)
		LabelCounts = {}
		for FeatVec in DataSet:
			Curr_Label = FeatVec[-1]
			if Curr_Label not in LabelCounts.keys():
				LabelCounts[Curr_Label] = 0
			LabelCounts[Curr_Label] += 1

		ShannonEnt = 0.0
		for key in LabelCounts:
			Prob = float(LabelCounts[key] / Samples_Num)
			ShannonEnt -= prob * log(prob,2)
		return ShannonEnt

	def ChooseBestFeatToSplit(self,DataSet):	#划分数据集（即挑出该分支上的最好筛选属性
		BaseEnt = CalShannonEnt(DataSet)
		Feats_Num = len(DataSet[0]) - 1
		BestInfoGain = 0.0
		BestFeat = 0

		for i in range(0,Feats_Num):
			FeatList = [temp[i] for temp in DataSet]
			Vals_Set = set(FeatList)
			NewEnt = 0.0

			for value in Vals_Set:
				SubDataSet = SplitDataSet(DataSet,i,value)
				Prob = len(SubDataSet)/float(len(DataSet))
				NewEnt += Prob * CalShannonEnt(SubDataSet)

			InfoGain = BaseEnt - NewEnt
			if InfoGain > BestInfoGain:
				BestInfoGain = InfoGain
				BestFeat = i 

		return BestFeat

	def MajorityCnt(self,ClassList):
		ClassCount = {}
		for vote in ClassList:
			if vote not in ClassCount.keys():
				ClassCount[vote] = 0
			ClassCount[vote] += 1

		SortedClassCount = sorted(ClassCount.iteritems(),key = operator.itemgetter(1),reversed = True)
		return SortedClassCount[0][0]

	def CreateTree(self,DataSet,Labels):
		ClassList = [temp[-1] for temp in DataSet]
		if len(DataSet[0]) == 1:
			return MajorityCnt(ClassList)

		BestFeat = ChooseBestFeatToSplit(DataSet)
		BestFeatLabel = Labels[BestFeat]
		Decistion_Tree = {BestFeatLabel:{}}
		del(Labels[BestFeat])

		FeatValues = [temp[BestFeat] for temp in DataSet]
		UniVals = Set(FeatValues)
		for Value in UniVals:
			SubLabels = Labels[:]
			SubDataSet = SplitDataSet(DataSet,BestFeat,Value)
			Decistion_Tree[BestFeatLabel][Value] = CreateTree(SubDataSet,SubLabels)

		return Decistion_Tree


	def Classify(Tree,FeatLabels, TestVec):
		FirstStr = Tree.keys()[0]
		SecondDict = Tree[FirstStr]
		FeatIndex = FeatLabels.index(FirstStr)
		for key in SecondDict.keys():
			if TestVec[FeatIndex] == key:
				if type(SecondDict[key]).__name__ == 'dict':
					ClassLabel = Classify(SecondDict[key],FeatLabels,TestVec)
				else :
					ClassLabel = SecondDict[key]
		return ClassLabel