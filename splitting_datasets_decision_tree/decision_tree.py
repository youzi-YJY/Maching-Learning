# -*- coding:utf-8 -*-
from math import log
import matplotlib.pyplot as plt
import operator
import pickle

decisionNode=dict(boxstyle="sawtooth",fc="0.8")#文本框类型为锯齿形
leafNode=dict(boxstyle="round4",fc="0.8")#叶节点为圆一点的四边形
arrow_args=dict(arrowstyle="<-")#箭头的类型


def createDataSet():
    dataSet=[[1,1,'yes'],
             [1,1,'yes'],
             [1,0,'no'],
             [0,1,'no'],
             [0,1,'no']]
    labels=['no surfacing','flippers']
    return dataSet,labels

def calcShannonEnt(dataSet):  # 计算数据的熵(entropy)
    numEntries=len(dataSet)  # 数据条数
    labelCounts={}#dictionary whose keys are the values in the final column
    for featVec in dataSet:
        currentLabel=featVec[-1] # 每行数据的最后一个字（类别）
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1  # 统计有多少个类以及每个类的数量
    shannonEnt=0.0
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries # 计算单个类的熵值
        shannonEnt-=prob*log(prob,2) # 累加每个类的熵值
    return shannonEnt

def splitDataSet(dataSet,axis,value): # 按某个特征分类后的数据
    '''
    :param dataSet:the data we'll split
    :param axis:the feature we’ll split on
    :param value:and the value of the feature to return
    Our dataset is a list of lists; you iterate over every item in the list and if it
    contains the value you’re looking for, you’ll add it to your newly created list. Inside the
    if statement, you cut out the feature that you split on.
    '''
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis]==value:
            reducedFeatVec =featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):  # 选择最优的分类特征
    numFeatures = len(dataSet[0])-1
    baseEntropy = calcShannonEnt(dataSet)  # 原始的熵
    bestInfoGain = 0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob =len(subDataSet)/float(len(dataSet))
            newEntropy +=prob*calcShannonEnt(subDataSet)  # 按特征分类后的熵
        infoGain = baseEntropy - newEntropy  # 原始熵与按特征分类后的熵的差值
        if (infoGain>bestInfoGain):   # 若按某特征划分后，熵值减少的最大，则此特征为最优分类特征
            bestInfoGain=infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    '''
    return the class that occurs with the greatest frequency.
    '''
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    classList=[example[-1] for example in dataSet]  # 类别：男或女
    if classList.count(classList[0])==len(classList):
        return classList[0]
    if len(dataSet[0])==1:
        return majorityCnt(classList)
    bestFeat=chooseBestFeatureToSplit(dataSet) #选择最优特征
    bestFeatLabel=labels[bestFeat]
    myTree={bestFeatLabel:{}} #分类结果以字典形式保存
    del(labels[bestFeat])#删除了最佳划分特征的标签
    featValues=[example[bestFeat] for example in dataSet]
    uniqueVals=set(featValues)
    for value in uniqueVals:
        subLabels=labels[:]
        myTree[bestFeatLabel][value]=createTree(splitDataSet\
                            (dataSet,bestFeat,value),subLabels)
    return myTree
'''
Attention：
If the value is a class label, then that child is a leaf node. If the value is another dictionary,
then that child node is a decision node and the format repeats itself.'''



def save_Tree(myTree):
    with open('savetree_file.txt','w') as f:
        f.write(str(myTree))

def load_Tree():
    with open('savetree_file.txt','r') as f:
        a=f.read()
        myTree=eval(a)
        return myTree

def getNumLeafs(myTree):#leaf node 叶子节点的个数
    '''
    Identifying the number of leaves in a tree and the depth
    getNumLeafs function only count the leaf nodes.
    getTreeDepth function stopping condition is the leaf node.
    '''
    #get out the first key and value, and the iterate over all the of the child nodes
    numsLeafs=0
    firstStr=list(myTree.keys())[0]
    secondDict=myTree[firstStr]
    #if the chile node is of type dict, it is another decision node and we should
    #recursively call the getNumLeafs function.
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            numsLeafs+=getNumLeafs(secondDict[key])
        else:
            numsLeafs+=1
    return numsLeafs

def getTreeDepth(myTree):#decision node 判断节点的个数
    '''
    To save the time and avoid make a tree from data every time.
    '''
    maxDepth=0
    firststr=list(myTree.keys())[0]
    secondDict=myTree[firststr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            thisDepth=1+getTreeDepth(secondDict[key])
        else:
            thisDepth=1
        if thisDepth>maxDepth: maxDepth=thisDepth
    return maxDepth


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def plotMidText(cntrPt, parentPt, txtString):
    '''
    calculate the midpoint between the parent and child nodes and puts a simple
    text in de middle.
    Plot text between child and parent
    '''
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


def plotTree(myTree, parentPt, nodeTxt):  # if the first key tells you what feat was split on
    numLeafs = getNumLeafs(myTree)  # this determines the x width of this tree
    depth= getTreeDepth(myTree)# the deepth of this tree.
    firstStr =list(myTree.keys())[0]  # the text label for this node should be this
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / numLeafs, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    #decrement the plotTree.yOff to make a note that be about to draw children nodes.
    plotTree.yOff = plotTree.yOff - 1.0 / depth
    #Go through the tree in a similar fashion as the getNumLeafs and getTreeDepth
    #If a node is a leaf node, you draw a leaf node.
    #If not, you recursively call plotTree again
    #Finally, after you finish plotting the child nodes, you increment the global Y offset.
    #Seperate the fistStr and seceondDict
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':  # test to see if the nodes are dictonaires, if not they are leaf nodes
            plotTree(secondDict[key], cntrPt, str(key))  # recursion
        else:  # it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0 / numLeafs
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 /depth


# if you do get a dictonary you know it's a tree, and the first element will be another dict

def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)  # no ticks
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()

def classify(inputTree,featLabels,textVec):
    firstStr=list(inputTree.keys())[0]
    secondDict=inputTree[firstStr]
    featIndex=featLabels.index(firstStr)
    #use index method to find out the first item in this list that matches firststr
    #with this in mind,you can recursively travel thr tree,comparing the values in testVec
    #to the values in tree.
    #Until you reach the node, you`ve made your classification and it`s time to exit.
    for key in secondDict.keys():
        if textVec[featIndex]==key:
            if type(secondDict[key]).__name__=='dict':
                classLabel=classify(secondDict[key],featLabels,textVec)
            else:
                classLabel=secondDict[key]
    return classLabel



def storeTree(inputTree,filename):
    fw=open(filename,'wb')
    pickle.dump(inputTree,fw)
    fw.close()

def garbTree(filename):
    fr=open(filename,'rb')
    return pickle.load(fr)

#以上测试过没有问题

if __name__=='__main__':
    dataSet,labels=createDataSet()
    myTree=createTree(dataSet,labels)
    #print(getTreeDepth(myTree))
    #print(getNumLeafs(myTree))
    createPlot(myTree)
    #因为createTree()函数中删除了最佳划分特征的标签，从而建立树以后需要再重新构建一次原始数据集.
    #dataSet,labels=createDataSet()
    #result=classify(myTree,labels,[1,1])
    storeTree(myTree,"decision_tree.txt")
    tree=garbTree("decision_tree.txt")
