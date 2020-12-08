#-*-coding:utf-8-*-
from numpy import *
import operator
import matplotlib.pyplot as plt
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


def file2martix(filename):
    '''
    datingTestSet.txt including:
    1000 entries
    Recorded the following features:
    one.Number of frequent flyer miles earned per year.
    two.Percentage of time spent playing video games.
    three.Liters of ice cream consumed per week.
    :param filename: datingTestSet
    :return: retrunMat,classLabelVector
    '''
    fr=open(filename)
    numberOfLines=len(fr.readlines())
    returnMat=zeros((numberOfLines,3))#生成numberOfLines*3全零矩阵
    classLabelVector=[]
    fr=open(filename)
    index=0
    for line in fr.readlines():
        line=line.strip()#剔除多余的空格
        listFromline=line.split('\t')#利用'\t'来分隔读取到的line
        returnMat[index,:]=listFromline[0:3]#take the first three elements and shove them into a row of matrix
        classLabelVector.append(listFromline[-1])#like the integer verison of the last item in the list
        index+=1
    return returnMat,classLabelVector

def plotpicture(dataingMata):
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(dataingMata[:,1],dataingMata[:,2])
    ax.axis([-2,25,-0.2,2.0])
    plt.xlabel('Percentage of Time Spent Playing Video Games')
    plt.ylabel('Listers of Ice Cream Consumed Per Week')
    plt.show()


def autoNorm(dataSet):
    '''
    Data-normalizing code
    :param dataSet: our data martix
    :return:normDataSet,ranges,minVals
    '''
    minVals=dataSet.min(0)#The 0 in dataSet.min(0) allows you to take the minimums from the columns
    maxVals=dataSet.max(0)#Same to the above
    #the shape of minVals and maxVals is 1*3 and our martix is 1000*3
    ranges=maxVals-minVals
    normDataSet=zeros(shape(dataSet))
    m=dataSet.shape[0]
    #tile function to create a martix the same size as our input martix and fill it up with many copies.
    normDataSet=dataSet-tile(minVals,(m,1))
    normDataSet=normDataSet/tile(ranges,(m,1))# / operator is element-wise division;linalg.solve(matA,matB) for martix division
    return normDataSet,ranges,minVals

def datingClassTest(normMat,datingLabels):
    '''
    Classifler testing code for dating site
    '''
    hoRatio=0.10
    m=normMat.shape[0]
    numTestVecs=int(m*hoRatio)
    errorCount=0.0
    for i in range(numTestVecs):
        classifierResult=classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classifier came back with: %s, the real answer is: %s"\
              %(classifierResult,datingLabels[i]))
        if (classifierResult!=datingLabels[i]):
            errorCount+=1.0
    print("the total error rate is: %f" %(errorCount/float(numTestVecs)))


def classifyPerson(datingLabels,normMat,minVals,ranges):
    '''
    Dating site predictor functions
    '''
    resultList=['not at all','in small does','in large doses']
    percenTats=float(input("percentage of time spent playing video games?"))
    ffMiles=float(input("frequent flier miles earned per year?"))
    iceCream=float(input("liters of ice cream consumed per year?"))
    inArr=array([ffMiles,percenTats,iceCream])
    classifierResult=classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print('You will probably like this person with: %s' %classifierResult)

dataMata,classLabelMat=file2martix('datingTestSet.txt')
plotpicture(dataMata)
#normDataSet,ranges,minVals=autoNorm(dataMata)
#datingClassTest(normDataSet,classLabelMat)
#classifyPerson(classLabelMat,normDataSet,minVals,ranges)
#以上测试过，没有问题.
