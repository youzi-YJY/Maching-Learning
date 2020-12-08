
from numpy import *
import matplotlib.pyplot as plt

def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    retArray=ones((shape(dataMatrix)[0],1))
    if threshIneq=='lt':
        retArray[dataMatrix[:,dimen]<=threshVal]=-1.0
    else:
        retArray[dataMatrix[:,dimen]>threshVal]=1.0
    return retArray

def buildStump(dataArr,classLabels,D):
    dataMartix=mat(dataArr)
    labelMat=mat(classLabels).T
    m,n=shape(dataMartix)
    numSteps=10.0
    bestStump={}
    bestClassEst=mat(zeros((m,1)))
    minError=inf
    for i in range(n):
        rangeMin=dataMartix[:,i].min()
        rangeMax=dataMartix[:,i].max()
        stepSize=(rangeMax-rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1):
            for inequal in ['lt','gt']:
                threshVal=(rangeMin+float(j)*stepSize)
                predictedVals=stumpClassify(dataMartix,i,threshVal,inequal)
            errArr=mat(ones((m,1)))
            errArr[predictedVals==labelMat]=0
            weightedError=D.T*errArr
            print("split:dim %d,thresh %.2f, thesh inequal:\
                    %s, the weighted error is %.3f" %\
                    (i,threshVal,inequal,weightedError))
            if weightedError<minError:
               minError=weightedError
               bestClassEst=predictedVals.copy()
               bestStump['dim']=i
               bestStump['thresh']=threshVal
               bestStump['ineq']=inequal
    return bestStump,minError,bestClassEst

def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    weakClassArr=[]
    m=shape(dataArr)[0]
    D=mat(ones((m,1))/m)
    #gives you the aggregate estimate of the class for every data point.
    aggClassEst=mat(zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst=buildStump(dataArr,classLabels,D)
        print("D",D.T)
        #tell the total classifier how much to weight the output from this output.
        alpha=float(0.5*log((1.0-error)/max(error,1e-16)))
        #This dictionary will contain all you need for classification.
        bestStump['alpha']=alpha
        weakClassArr.append(bestStump)
        print("classEst:",classEst.T)
        #calculate the new weights D for the next iteration.
        expon=multiply(-1*alpha*mat(classLabels).T,classEst)
        D=multiply(D,exp(expon))
        D=D/D.sum()
        #aggregate error calculation.
        aggClassEst+=alpha*classEst
        print("aggClassEst",aggClassEst.T)
        #To get the binary class use the sign() function.
        aggErrors=multiply(sign(aggClassEst)!=mat(classLabels).T,ones((m,1)))
        errorRate=aggErrors.sum()/m
        print("total error:",errorRate,"\n")
        if error==0.0:
            break
    return weakClassArr,aggClassEst

def adaClassify(dataToClass,classifierArray):
    dataMatrix=mat(dataToClass)
    m=shape(dataMatrix)[0]
    aggClassEst=mat(zeros((m,1)))
    for i in range(len(classifierArray)):
        classEst=stumpClassify(dataMatrix,classifierArray[i]['dim'],\
                               classifierArray[i]['thresh'],\
                               classifierArray[i]['ineq'])
        aggClassEst+=classifierArray[i]['alpha']*classEst
        print(aggClassEst)
    return sign(aggClassEst)

def loadData(filename):
    numFeat=len(open(filename).readline().split('\t'))
    dataMat=[]
    labelMat=[]
    fr=open(filename)
    for line in fr.readlines():
        lineArr=[]
        curline=line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curline[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curline[-1]))
    return dataMat,labelMat


def plotRoc(predStrengths,classLabels):
    cur=(1.0,1.0)
    ySum=0.0
    numPosClas=sum(array(classLabels)==1.0)
    yStep=1/float(numPosClas)
    xStep=1/float(len(classLabels)-numPosClas)
    sortedIndicies=predStrengths.argsort()
    fig=plt.figure()
    fig.clf()
    ax=plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index]==1.0:
            delX=0
            delY=yStep
        else:
            delX=xStep
            delY=0
            ySum+=cur[1]
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY],c='b')
        cur=(cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Roc curve for AdaBoost Horse Colic Detection System')
    ax.axis([0,1,0,1])
    plt.show()
    print("The area Under the Curve is: ",ySum*xStep)

if __name__=='__main__':
    datArr,labelArr=loadData('horseColicTraining.txt')
    classifierArray,aggClassEst= adaBoostTrainDS(datArr, labelArr, 10)
    #print(adaBoostTrainDS(datArr,labelArr,10))
    #testArr,testLabelArr=loadData('horseColicTest.txt')
    #print(adaClassify(testArr,classifierArray))
    plotRoc(aggClassEst.T,labelArr)