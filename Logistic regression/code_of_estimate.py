# -*-coding:utf-8 -*-
from numpy import *
from math import *


def sigmoid(inX):
    try:
        return 1.0 / (1 + exp(-inX))
    except OverflowError:
        return float('inf')

def stocGradAscent1(dataMartix,classLabel,numIter=150):
    '''
    Two things have been improved
    alpha changes on each iteration
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

def classifyVector(inX,weights):
    prob=sigmoid(sum(inX*weights))
    if prob>0.5:return 1.0
    else: return 0.0

def colicTest():
    frTrain=open('data/horseColicTraining.txt')
    frtest=open('data/horseColicTest.txt')
    trainingSet=[]
    trainingLabels=[]
    for line in frTrain.readlines():
        currline=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currline[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currline[21]))
    trainWeights=stocGradAscent1(array(trainingSet),trainingLabels,500)
    errcount=0
    numTestVec=0.0
    for line in frtest.readlines():
        numTestVec+=1.0
        currline=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currline[i]))
        if int(classifyVector(array(lineArr),trainWeights))!=int(currline[21]):
            errcount+=1
    errRate=(float(errcount)/numTestVec)
    print("The error rate of this test is: %f" %errRate)
    return errRate

def multiTest():
    numTests=10
    errorSum=0.0
    for k in range(numTests):
        errorSum+=colicTest()
    print("After %d iterations the average error rate is: %f" %(numTests,errorSum/float(numTests)))

if __name__=='__main__':
    multiTest()