from numpy import *
import matplotlib.pyplot as plt
def loadSimpData():
    datMat=mat([[1.,2.1],
                [2.,1.1],
                [1.3,1.],
                [1.,1.],
                [2.,1]])
    classLabels=[1.0,1.0,-1.0,-1.0,1.0]
    return datMat,classLabels

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
            #This is the line where AdaBoost interacts with the classifier. You’re
            #evaluating your classifier based on the weights D, not on another error measure. If you
            #want to use another classifier, you’d need to include this calculation to define the best
            #classifier for D.
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

def adaClassify(datToClass,classifierArr):
    dataMartix=mat(datToClass)
    m=shape(dataMartix)[0]
    aggClassEst=mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst=stumpClassify(dataMartix,classifierArr[i]['dim'],\
                               classifierArr[i]['thresh'],\
                               classifierArr[i]['ineq'])
        aggClassEst+=classifierArr[i]['alpha']*classEst
        print(aggClassEst)
    return sign(aggClassEst)

def loadDataSet(fileName):
    numFeat=len(open(fileName).readline().split('\t'))
    dataMat=[]
    labelMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def plotROC(predStrengths,classLabels):
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
    plt.title('ROC curve for AdaBoost Horse Colic Detection System')
    ax.axis([0,1,0,1])
    plt.show()
    print("the Area Under the Curve is:",ySum*xStep)



if __name__=='__main__':
    #D=mat(ones((5,1))/5)
    #dataMat,classLabel=loadSimpData()
    #bestStump,minError,bestClassEst=buildStump(dataMat,classLabel,D)
    #classifierArr=adaBoostTrainDS(dataMat,classLabel,30)
    #result=adaClassify([0,0],classifierArr)
    #result=adaClassify([[5,5],[0,0]],classifierArr)
    dataArr,labelArr=loadDataSet('horseColicTraining.txt')
    classifierArr,aggClassEst=adaBoostTrainDS(dataArr,labelArr,10)
    #testArr,testLabelArr=loadDataSet('horseColicTest.txt')
    #predict10=adaClassify(testArr,classifierArr)
    #errArr=mat(ones((67,1)))
    #print(errArr[predict10!=mat(testLabelArr).T].sum())
    plotROC(aggClassEst.T,labelArr)