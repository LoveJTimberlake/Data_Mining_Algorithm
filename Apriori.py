#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 20:52:16 2018

@author: justintimberlake
"""

import numpy as np 


#适用于数值型数据或标称型数据
#频繁项集与关联规则 支持度与置信度
#关联规则中只要满足其中的子集，那么就可以给其推荐剩下的子项

'''
过程：
1.收集数据
2.用集合保存数据并进行预处理
3.分析数据 
4.用Apriori找出频繁项集
5.测试
6.从频繁项集找出关联规则
'''

#发现频繁项集

def loadDataSet(filename):      #数据集中只有行为数据而没有标签/y值
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        data = line.strip().split('\t')
        dataMat.append(data)
    fr.close()
    return dataMat

def createC1(dataSet):
    C1 = []     #因为后面的项集都是以嵌套列表的方式存储，所以从单项开始就要用列表存储
    for transaction in dataSet:
        for item in transaction:
            if [item] not in C1:
                C1.append([item])
    C1.sort()
    return map(frozenset,C1)    #Ck都是不可变的集合set

'''
et无序排序且不重复，是可变的，有add（），remove（）等方法。既然是可变的，所以它不存在哈希值。基本功能包括关系测试和消除重复元素. 
集合对象还支持union(联合), intersection(交集), difference(差集)和sysmmetric difference(对称差集)等数学运算. 
sets 支持 x in set, len(set),和 for x in set。作为一个无序的集合，sets不记录元素位置或者插入点。
因此，sets不支持 indexing, 或其它类序列的操作。
frozenset是冻结的集合，它是不可变的，存在哈希值，好处是它可以作为字典的key，也可以作为其它集合的元素。
缺点是一旦创建便不能更改，没有add，remove方法。
'''

def scanD(D,Ck,minSupport): #Ck是候选集
    ssCnt = {}  #项集的支持度
    for tid in D:
        for can in Ck:  #长度为k的候选集
            if can.issubset(tid):
                if can not in ssCnt.keys():
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = float(len(D))    #事务总数
    retList = []    #当前满足最小支持度的项集，是频繁项集
    supportData = {}    #所有出现过的
    for key,value in ssCnt.items():
        support = value/numItems
        if support >= minSupport:
            retList.append(key)
        supportData[key] = support #doubts maybe should be one more tab
    return retList,supportData 

def aprioriGen(Lk,k):   #k是下一步要生成的项集长度
    retList = [] 
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1,lenLk):
            L1 = list(Lk[i])[:k-2]  #当前项集长度为k-1,要生成k长度的新项集，只要两个旧项集的前k-2项([:k-2])一样，那么即可将这两个旧的融合在一起形成新的
            L2 = list(Lk[j])[:k-2]  
            L1.sort()
            L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])   #集合的或（并）运算
    return retList

def apriori(dataSet,minSupport = 0.5):
    C1 = createC1(dataSet)
    D = map(set,dataSet)    #将每个事务中重复的item去除，将当前数组变为set
    L1,supportdata = scanD(D,C1,minSupport)
    L = [L1]    #将所有迭代获得的项集都放在一个列表中
    k = 2
    while(len(L[k-2]) > 0):     #因为1-项集在L中索引为0 所以新的项集长度k是从索引为k-2的(k-1)-项集中获得的
        Ck = aprioriGen(L[k-2],k)
        Lk,SupK = scanD(D,Ck,minSupport)
        supportdata.update(SupK)    #将新的字典加入到旧的字典中，若有重复的，则是直接覆盖那项
        L.append(Lk)
        k += 1
    return L,supportdata

#生成关联规则
def generateRules(L,supportData,minConf = 0.7):
    bigRuleList = []    #最终规则存放处,要根据可信度对其排序
    for i in range(1,len(L)):   #len(L)是所有频繁项集的个数，只获取k-频繁项集 (k > 1)
        for freqSet in L[i]:    #freqSet是每个k-频繁项
            H1 = [frozenset([item]) for item in freqSet]    #H1是每个k-频繁项的单项集合列表
            if i> 1:    #当频繁项集的个数不止一个且现在是第二三个的频繁项集时，就要考虑合并多个频繁项集
                rulesFromConseq(freqSet,H1,supportData,bigRuleList,minConf)
            else:       #对于2-频繁项集，直接计算其可信度
                calcConf(freqSet,H1,supportData,bigRuleList,minConf)    
    return bigRuleList

def calcConf(freqSet,H,supportData,brl,minConf = 0.7):
    prunedH = [] 
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet - conseq]   #conseq是单项，从左边原来的全集合慢慢循环剪掉单项并将其加至右边
        if conf >= minConf:
            print(freqSet - conseq + '-->' + conseq +' conf:' + conf)
            brl.append((freqSet-conseq,conseq,conf))
            prunedH.append(conseq)
    return prunedH

def rulesFromConseq(freqSet,H,supportData,brl,minConf = 0.7):
    m = len(H[0])   #单项长度
    if(len(freqSet) > (m+1)):   #当一个k-频繁单项的元素个数k大于单项长度+1时，则可以移除长度为m的项集
        Hmp1 = aprioriGen(H,m+1)    #对于H的k+1-候选项集
        Hmp1 = calcConf(freqSet,Hmp1,supportData,brl,minConf)
        if(len(Hmp1) > 1):
            rulesFromConseq(freqSet,Hmp1,supportData,brl,minConf)


#测试代码
ds = loadDataSet('train.txt')
L,supportData = apriori(ds)
rules = generateRules(L,supportData,minConf = 0.7)
print(rules)
rules = generateRules(L,supportData,minConf = 0.5)
print(rules)




















