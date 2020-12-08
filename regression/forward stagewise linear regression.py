from numpy import *
import  matplotlib.pyplot as plt


def loadData(filename):
    numFeat=len(open(filename).readline().split('\t'))-1
    dataMat=[]
    labelMat=[]
    fr=open(filename)
    for line in fr.readlines():
        lineArr=[]
        curline=line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curline[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curline[-1]))
    return dataMat,labelMat


def rssError(yArr,yHatArr):
    return ((yArr-yHatArr)**2).sum()

def regularize(xMat):#regularize by columns
    intMat=xMat.copy()
    inMeans=mean(intMat,0)
    inVar=var(intMat,0)
    intMat=(intMat-inMeans)/inVar
    return intMat

def stagewise(xArr,yArr,eps=0.01,numIt=100):
    xMat=mat(xArr);yMat=mat(yArr).T
    yMean=mean(yMat,0)
    yMat=yMat-yMean
    xMat=regularize(xMat)
    m,n=shape(xMat)
    ws=zeros((n,1));wsTest=ws.copy;wsMax=ws.copy()
    for i in range(numIt):
        print(ws.T)
        lowestError=inf
        for j in range(n):
            for sign in [-1,1]:
                wsTest=ws.copy()
                wsTest[j]+=eps*sign
                yTest=xMat*wsTest
                rssE=rssError(yMat.A,yTest.A)
                if rssE<lowestError:
                    lowestError=rssE
                    wsMax=wsTest
        ws=wsMax.copy()
        #returnMat[i,:]=ws.T
    #return returnMat



if __name__=='__main__':
    abX,abY=loadData('data/abalone.txt')
    stagewise(abX,abY,0.001,200)
