from math import *
from numpy import *
import matplotlib.pyplot as plt
import random

def loadDataSet():
    dataMat=[]
    labelMat=[]
    fr=open('data/testSet.txt')
    for line in fr.readlines():
        lineArr=line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

# dataMatIn存放的是3个特征，是100*3的矩阵
# classLabels存放的是类别标签，是1*100的行向量
def gradAscent(dataMatIn,classLabels):
    #转换为numpy类型的矩阵数据类型
    dataMatrix=mat(dataMatIn)
    labelMat=mat(classLabels).transpose()
    m,n=shape(dataMatrix)
    #m=100 n=3
    #while step size is 1 will overflow
    alpha=0.001#步长
    maxCycles=500#迭代次数
    weights=ones((n,1))#n*1全1矩阵
    for k in range(maxCycles):
        #矩阵运算
        h=sigmoid(dataMatrix*weights)#列向量的个数等于样本的数目
        error=(labelMat-h)#计算相对错误率
        weights=weights+alpha*dataMatrix.transpose()*error
    #getA() change a numpy array into a normal array.
    return weights.getA()


def plotBestFit(weights):
    dataMat,labelMat=loadDataSet()
    dataArr=array(dataMat)
    n=shape(dataArr)[0]
    xcord1=[];ycord1=[]
    xcord2=[];ycord2=[]
    for i in range(n):
        if int(labelMat[i])==1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x=arange(-3.0, 3.0, 0.1)
    ##diffcult to understand
    y=(-weights[0]-weights[1]*x)/weights[2]#最佳拟合直线
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

def stoGradAscent0(dataMartix,classLabels):
    '''
    This function is similar to gradient ascent except that
    the variables h and error are now single values rather than vectors.
    There is also no matrix conversion , so all of the variables are Numpy arrays.
    '''
    m,n=shape(dataMartix)
    alpha=0.01
    weights=ones(n)
    for i in range(m):
        h=sigmoid(sum(dataMartix[i]*weights))
        error=classLabels[i]-h
        weights=weights+alpha*error*dataMartix[i]
    return weights

def stocGradAscent1(dataMartix,classLabel,numIter=150):
    '''
    Two things have been improved
    alpha changes on each iteration
    Random Gradient ascent function.
    '''
    m,n=shape(dataMartix)
    weights=ones(n)
    dataIndex=range(m)
    for j in range(numIter):
        for i in range(m):
            alpha=4/(1.0+i+j)+0.01
            randIndex=int(random.uniform(0,len(dataIndex)))
            h=sigmoid(sum(dataMartix[randIndex]*weights))
            error=classLabel[randIndex]-h
            weights=weights+alpha*error*dataMartix[randIndex]
            #del(dataMartix[randIndex])
    return weights

#以上测试过，没有问题。

if __name__=='__main__':
    dataArr,labelMat=loadDataSet()
    result=gradAscent(dataArr,labelMat)
    plotBestFit(result)
    weights=stocGradAscent1(array(dataArr),labelMat,500)
    plotBestFit(weights)


