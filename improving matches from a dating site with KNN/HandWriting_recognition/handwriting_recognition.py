#-*- coding:utf-8 -*-
from numpy import *
from os import listdir
import operator

def classify0(inX,dataSet,labels,k):
    '''
    function would like this:
    For very point n our dataset:
    calculate the distance between inX and current point
    sort the distance in increasing order
    take k items with lowest disrtance to inX
    find the majority calss among these items
    return the majority class as our prediction for the class of inX
    :param inX: the input vector to classify called inX
    :param dataSet: our full martix of training examples
    :param labels:  a vector of labels
    :param k: the number of nearest neighbors to use in the voting
    :return: sortedClassCount[0][0]
    '''
    dataSetsize=dataSet.shape[0]#记录数组第一维的大小
    #欧几里得距离的公式体现
    diffMat=tile(inX,(dataSetsize,1))-dataSet
    sqDiffMat=diffMat**2
    sqDistance=sqDiffMat.sum(axis=1)#axis=1 列求和
    distance=sqDistance**0.5#开方
    sortedDistIndicies=distance.argsort()#元素从小到大排序 提取对对应的index,默认升序排列.
    classCount={}
    for i in range(k):#the input k should always be a positive integer
        voteIlabel=labels[sortedDistIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    #items返回的是列表对象.从Python3.5开始使用
    #using the itemgetter method from the operator module imported in the second line of the program
    #operator.itemgetter函数获取的不是值，而是定义了一个函数，通过该函数作用到对象上才能获取值
    #sorted(iterable[, cmp[, key[, reverse]]])
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def img2vevtor(filename):
    '''
    The function creates a 1x1024 NumPy array, then opens the given
    file, loops over the first 32 lines in the file, and stores the integer value of the first 32
    characters on each line in the NumPy array. This array is finally returned.
    '''
    returnVect=zeros((1,1024))
    fr=open(filename)
    for i in range(32):
        lineStr=fr.readline()
        for j in range(32):
            returnVect[0,32*i+j]=int(lineStr[j])
    return returnVect

def handwritigClassTest():
    hwLables=[]
    trainingFileList=listdir('trainingDigits')
    m=len(trainingFileList)#1934
    trainingMat=zeros((m,1024))
    for i in range(m):
        fileNameStr=trainingFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        hwLables.append(classNumStr)
        trainingMat[i,:]=img2vevtor('trainingDigits/%s' %fileNameStr)

    testFileList=listdir('testDigits')
    errcount=0.0
    mTest=len(testFileList)
    for i in range(mTest):
        fileNameStr=testFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        vectorUnderTest=img2vevtor('testDigits/%s' %fileNameStr)
        classifierResult=classify0(vectorUnderTest,trainingMat,hwLables,3)
        print("the classifier came back with: %d,the real answer is %d"\
              %(classifierResult,classNumStr))
        if (classifierResult!=classNumStr):
            errcount+=1
    print("\n the total number of errors is: %d" %errcount)
    print("\n thr total error rate is: %f" %(errcount/float(mTest)))

handwritigClassTest()