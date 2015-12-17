import dtree
import treeplotter
def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__=='dict':
                classLabel = classify(secondDict[key],featLabels,testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


myDat,labels = dtree.createDataSet()
print(labels)
myTree = treeplotter.retrieveTree(0)
print(myTree)
print('classify(myTree,labels,[1,0]):',classify(myTree,labels,[1,0]))
print('classify(myTree,labels,[1,1]):',classify(myTree,labels,[1,1]))
