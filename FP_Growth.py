#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 12:41:17 2018

@author: justintimberlake
"""



#构建FP树
#从FP树中挖掘频繁项集

class TreeNode:
    def __init__(self,nameValue,numOccur,parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None    #连接同个元素的不同结点
        self.parent = parentNode
        self.children = {} 
        
    def inc(self,numOccur):
        self.count += numOccur
    
    def disp(self,ind = 1):
        print(' ' * ind + self.name + ' ' + self.count)
        for child in self.children:
            child.disp(ind+1)
    
def createTree(dataSet,minSup = 1):
    headerTable = {}    
    for trans in dataSet:   #创建头表
        for item in trans:
            headerTable[item] = headerTable.get(item,0) + dataSet[trans] #doubt
    for k in headerTable.keys():    #headerTable中的都是1-频繁项
        if headerTable[k] < minSup:
            del(headerTable[k])
    
    freqItemSet = set(headerTable.keys())
    if len(freqItemSet) == 0:
        return None,None 
    
    for k in headerTable.keys():
        headerTable[k] = [headerTable[k],None]
    
    retTree = TreeNode('Null Set', 1, None)
    for tranSet, count in dataSet.items():  #生成树
        localD = {} 
        for item in tranSet:
            if item in freqItemSet:
                localD[item] = headerTable[item][0] #localD[item]中装的是当前事务中该item的出现次数
        if len(localD) > 0: #当有频繁单项时
            orderedItems = [v[0] for v in sorted(localD.items,key = lambda p : p[1], reverse = True)]   #从大至小
            updateTree(orderedItems,retTree,headerTable,count)
    return retTree, headerTable

def updateTree(items,inTree,headerTable,count):
    if items[0] in inTree.children:
        inTree.children[items[0]].inc(count)
    else:
        inTree.children[items[0]] = TreeNode(items[0],count,inTree)
        if headerTable[items[0]][1] == None:
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1],inTree.children[items[0]])
    if len(items) > 1:  #递归调用
        updateTree(items[1:],inTree.children[items[0]], headerTable,count)
        
def updateHeader(nodeToTest, targetNode):
    while(nodeToTest.nodeLink != None):
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode

def loadSimpleMat():
    simpleMat = [['r','z','h','j','p'],
                 ['z','y','x','w','v','u','t','s'],
                 ['z'],
                 ['r','x','n','o','s'],
                 ['y','r','x','z','q','t','p'],
                 ['y','z','x','e','q','s','t','m']]
    return simpleMat

def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1   #输入的数据就是 trans count
    return retDict
            


#抽取条件模式基
    
#找到以给定元素结尾的所有路径的函数
def ascendTree(leafNode,prefixPath):    #递归向上将结点加进去
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)

def findPrefixPath(basePat, treeNode):  #basePat表示频繁单项 treeNode为当前要搜索的结点
    condPats = {} 
    while treeNode != None:
        prefixPath = [] 
        ascendTree(treeNode,prefixPath)
        if len(prefixPath ) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink    #变成同个元素在不同路径上的点
    return condPats

def mineTree(inTree,headerTable,minSup,preFix, freqItemList):
    bigL = [v[0] for v in sorted(headerTable.items(), key = lambda p : p[1])]   #按频数排序后的频繁单项
    
    for basePat in bigL:
        newFreqSet = preFix.copy() 
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        condPattBases = findPrefixPath(basePat,headerTable[basePat][1]) #headerTable[basePat][1]是那个link指向的从左到右第一个结点
        myCondTree, myHead = createTree(condPattBases,minSup)
        
        if myHead != None:
            mineTree(myCondTree,myHead,minSup,newFreqSet,freqItemList)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            