from numpy import *
import matplotlib.pyplot as plt
from math import *

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

def ridgeRegres(xMat,yMat,lam=0.2):
    xTx=xMat.T*xMat
    denom=xTx+eye(shape(xMat)[1])*lam
    if linalg.det(denom)==0.0:
        print("This matrix is singular,cannot do inverse")
        return
    ws=denom.I*(xMat.T*yMat)
    return ws


def ridgeTest(xArr,yArr):
    xMat=mat(xArr)
    yMat=mat(yArr).T
    #axis 不设置值，对 m*n 个数求均值，返回一个实数
    #axis = 0：压缩行，对各列求均值，返回 1* n 矩阵
    #axis =1 ：压缩列，对各行求均值，返回 m *1 矩阵
    yMean=mean(yMat,0)
    yMat=yMat-yMean
    xMeans=mean(xMat,0)
    xVar=var(xMat,0)
    xMat=(xMat-xMeans)/xVar
    numTestPts=30
    wMat=zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws=ridgeRegres(xMat,yMat,exp(i-10))
        wMat[i,:]=ws.T
    return wMat


def plotfigure(weights):
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(weights)
    plt.show()

if __name__=='__main__':
    abX,abY=loadData('data/abalone.txt')
    ridgeWeights=ridgeTest(abX,abY)
    plotfigure(ridgeWeights)