from numpy import *
import matplotlib.pyplot as plt

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

#Standard regression function

def standRegres(xArr,yArr):
    xMat=mat(xArr)
    yMat=mat(yArr).T
    xTx=xMat.T*xMat
    #Check the xTx martix if it can calculate inverse.
    #Compute the determinant of array and check.
    if linalg.det(xTx)==0.0:
        print("This martix is singular,cannot do inverse")
        return
    ws=xTx.I*(xMat.T*yMat)
    #ws = linalg.solve(xTx,xMat.T*yMatT)
    return ws

#Locally weighted linear regression function

def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat=mat(xArr)
    yMat=mat(yArr).T
    m=shape(xMat)[0]
    #create the diagonal martix weights.
    weights=mat(eye(m))
    for j in range(m):
        diffmat=testPoint-xMat[j,:]
        #only change the diagonal element.
        weights[j,j]=exp(diffmat*diffmat.T/(-2.0*k**2))
    xTx=xMat.T*(weights*xMat)
    if linalg.det(xTx)==0.0:
        print("This martix is singular,cannot do inverse")
        return
    #inverse
    ws=xTx.I*(xMat.T*(weights*yMat))
    return testPoint*ws



def lwlrTest(testArr,xArr,yArr,k=1.0):
    m=shape(testArr)[0]
    yHat=zeros(m)
    for i in range(m):
        yHat[i]=lwlr(testArr[i],xArr,yArr,k)
    return yHat

'''
def pltshow(xMat,yArr):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 1].flatten().A[0], mat(yArr).T.flatten().A[0], s=2, c='red')
    xCopy=xMat.copy()
    strInd = xCopy.sort(0)
    Xsort = xMat[strInd][:, 0, :]
    ax.plot(Xsort[:, 1], yHat[strInd])
    plt.show()

def pltshow(xMat,yMat,ws):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat2 = xCopy * ws
    ax.plot(xCopy[:, 1], yHat2)
    plt.show()'''

if __name__=='__main__':
    xArr,yArr=loadData('data/ex0.txt')
    #ws=standRegres(xArr,yArr)
    #yHat=lwlrTest(xArr,xArr,yArr,1.0)
    #xMat=mat(xArr)
    #yMat=mat(yArr)
    #yHat=xMat*ws
    #result=corrcoef(yHat.T,yMat)

