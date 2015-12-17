# python机器学习之 K-邻近算法

@简单的理解：[ 采用测量不同特征值之间的距离方法进行分类 ]

- **优点** ：精度高、对异常值不敏感、无数据输入假定
 
- **缺点** ：计算复杂度高，空间复杂度高；
- **适应数据范围** ：数值型、标称型；

-------------------
[TOC]
## kNN简介

> **kNN 原理** ：存在一个样本数据集合，也称作训练集或者样本集，并且样本集中每个数据都存在标签，即样本集实际上是 **每条数据** 与 **所属分类** 的 **对应关系**。
 > **核心思想** ：若输入的数据没有标签，则新数据的每个特征与样本集中数据对应的特征进行比较，该算法提取样本集中特征最相似数据（最近邻）的分类标签。
 > **k** ：选自最相似的k个数据，通常是不大于20的整数，最后选择这k个数据中出现次数最多的分类，作为新数据的分类。


### k-近邻算法的一般流程
```sequence

1.收集数据：可以使用任何方法。
2.准备数据：距离计算所需的数值，最好是结构化的数据格式。
3.分析数据：可以使用任何方法。
4.训练算法：此不走不适用于k-近邻算法。
5.测试算法：计算错误率。
6.使用算法：首先需要输入样本数据和结构化的输出结果，然后运行k-近邻算法判定输入数据分别属于哪个分类，最后应用对计算出的分类之行后续的处理。
```
###example1
#### python导入数据
```	python
from numpy import *
import operator

def	createDataSet():
	group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
	labels = ['A','A','B','B']
	return group,labels
```
#### python处理数据
``` python 
#计算已知类别数据集中的点与当前点之间的距离（欧式距离）
#按照距离递增次序排序
#选取与当前点距离最小的K个点
#确定前K个点所在类别的出现频率
#返回前k个点出现频率最高的类别最为当前点的预测分类
#inX输入向量，训练集dataSet,标签向量labels，k表示用于选择最近邻的数目
def	clissfy0(inX,dataSet,labels,k):
	dataSetSize = dataSet.shape[0]
	diffMat = tile(inX,(dataSetSize,1)) - dataSet
	sqDiffMat = diffMat ** 0.5
	sqDistances = sqDiffMat.sum(axis=1)
	distances = sqDistances ** 0.5
	sortedDistIndicies = distances.argsort()
	classCount = {}
	for i in range(k):
		voteLabel = labels[sortedDistIndicies[i]]
		classCount[voteLabel] = classCount.get(voteLabel,0) + 1
	sortedClassCount = sorted(classCount.iteritems(),
		key = operator.itemgetter(1),reverse = True)
	return sortedClassCount[0][0]
```
####python数据测试
```python
import kNN
from numpy import *

dataSet,labels = createDataSet()
testX = array([1.2,1.1])
k = 3
outputLabelX = classify0(testX,dataSet,labels,k)
testY = array([0.1,0.3])
outputLabelY = classify0(testY,dataSet,labels,k)

print('input is :',testX,'output class is :',outputLabelX)
print('input is :',testY,'output class is :',outputLabelY)
```
####python结果输出
```
('input is :', array([ 1.2,  1.1]), 'output class is :', 'A')
('input is :', array([ 0.1,  0.3]), 'output class is :', 'B')
```
###example2使用k-近邻算法改进约会网站的配对效果
#### 处理步骤
```
1.收集数据：提供文本文件
2.准备数据：使用python解析文本文件
3.分析数据:使用matplotlib画二维扩散图
4.训练算法：此步骤不适用与k－近邻算法
5.测试算法：使用提供的部份数据作为测试样本
6:使用算法：输入一些特征数据以判断对方是否为自己喜欢的类型
```
####python 整体实现
```python
#coding:utf-8
from numpy import *
import operator
from kNN import classify0
import matplotlib.pyplot as plt

def file2matrmix(filename):
    fr = open(filename)
    arrayLines = fr.readlines()
    numberOfLines = len(arrayLines)
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index +=1

    return returnMat,classLabelVector

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))

    return normDataSet,ranges,minVals

def datingClassTest():
    hoRatio = 0.10
    datingDataMat,datingLabels = file2matrmix('datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print('the classifier came back with: %d, the real answer is: %d' %(classifierResult,datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print('the total error rate is: %f' %(errorCount / float(numTestVecs)))

def classifyPerson():
    resultList = ['not at all','in small doses','in large doses']
    percentTats = float(raw_input('percentage of time spent playing video games?'))
    ffMiles = float(raw_input('frequent flier miles earned per year?'))
    iceCream = float(raw_input('liters of ice cream consumed per year?'))
    datingDataMat,datingLabels = file2matrmix('datingTestSet2.txt')
    normMat,ranges,minVals =autoNorm(datingDataMat)
    inArr = array([ffMiles,percentTats,iceCream])
    classifierResult = classify0((inArr - minVals) / ranges,normMat,datingLabels,3)
    print('you will probably like this person:',resultList[classifierResult - 1])

datingDataMat,datingLabels = file2matrmix('datingTestSet2.txt')
classifyPerson()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0 * array(datingLabels),15.0 * array(datingLabels))
plt.show()
```
###K-最近邻算法总结
>**k近邻算法**是最简单有效的分类算法，必须全部保存全部数据集，如果训练数据集很大，必须使用大量的存储空间，同时由于必须对数据集中的每个数据计算距离值，实际使用可能非常耗时。
>**k近邻算法**无法给出任何数据的基础结构信息，我们无法知晓平均实例样本和典型实例样本具有神秘特征。
