#coding:utf-8
#kNN :K最近邻算法

from numpy import *
import operator

def createDataSet():
    group = array([(1.0,0.9),[1.0,1.0],[0.1,0.2],[0.0,0.1]])
    labels = ['A','A','B','B']

    return group,labels

def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX,(dataSetSize,1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistIndicies[i]]
        classCount[voteLabel] = classCount.get(voteLabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(),
        key = operator.itemgetter(1),reverse = True)

    return sortedClassCount[0][0]

#test
dataSet,labels = createDataSet()

testX = array([1.2,1.1])
k = 3
outputLabelX = classify0(testX,dataSet,labels,k)
testY = array([0.1,0.3])
outputLabelY = classify0(testY,dataSet,labels,k)

print('input is :',testX,'output class is :',outputLabelX)
print('input is :',testY,'output class is :',outputLabelY)
