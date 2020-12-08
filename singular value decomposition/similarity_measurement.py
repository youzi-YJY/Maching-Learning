from numpy import *
from numpy import linalg as la

def loadExData():
    return[[1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [1, 1, 1, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1]]

def ecludSim(inA,inB):
    return 1.0/(1.0+la.norm(inA-inB))

def pearsSim(inA,inB):
    if len(inA)<3:
        return 1.0
    return 0.5+0.5*corrcoef(inA,inB,rowvar=0)[0][1]

def cosSim(inA,inB):
    num=float(inA.T*inB)
    denom=la.norm(inA)*la.norm(inB)
    return 0.5+0.5*(num/denom)

def stanEst(dataMat,user,simMeas,item):
    n=shape(dataMat)[1]
    simTotal=0.0; ratSimTotal=0.0
    for j in range(n):
        userRating=dataMat[user,j]
        if userRating==0:continue
        overLap=nonzero(logical_and(dataMat[:,item].A>0,\
                                    dataMat[:,j].A>0))[0]
        if len(overLap)==0:similarity=0
        else:similarity=simMeas(dataMat[overLap,item],\
                                dataMat[overLap,j])
        simTotal+=similarity
        ratSimTotal+=similarity*userRating
    if simTotal==0:
        return 0
    else:return ratSimTotal/simTotal

def recommend(dataMat,user,N=3,simMeas=cosSim,estMethod=stanEst):
    unratedItems=nonzero(dataMat[user,:].A==0)[1]
    if len(unratedItems)==0:return 'you rated everything'
    itemScores=[]
    for item in unratedItems:
        estimatedScore=estMethod(dataMat,user,simMeas,item)
        itemScores.append((item,estimatedScore))
    return sorted(itemScores,key=lambda jj:jj[1],reverse=True)[:N]

def svdEst(dataMat,user,simMeas,item):
    n=shape(dataMat)[1]
    simTotal=0.0; ratSimTotal=0.0
    U,Sigma,VT=la.svd(dataMat)
    Sig4=mat(eye(4)*Sigma[:4])
    xformedItems=dataMat.T*U[:,:4]*Sig4.I
    for j in range(n):
        userRating=dataMat[user,j]
        if userRating==0 or j==item:continue
        similarity=simMeas(xformedItems[item,:].T,\
                           xformedItems[j,:].T)
        print('the %d and %d similarity is: %f' %(item,j,similarity))
        simTotal+=similarity
        ratSimTotal+=similarity*userRating
    if simTotal==0:return 0
    else:return ratSimTotal/simTotal

#Image-compression functions
def printMat(inMat,thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i,k])>thresh:
                print(1,)
            else:
                print(0,)
        print('')

def imagePress(numSV=3,thresh=0.8):
    myl=[]
    for line in open('0_5.txt').readlines():
        newRow=[]
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat=mat(myl)
    print("*****Original matrix*****")
    printMat(myMat,thresh)
    U,Sigma,VT=la.svd(myMat)
    SigRecon=mat(zeros((numSV,numSV)))
    for k in range(numSV):
        SigRecon[k,k]=Sigma[k]
    reconMat=U[:,:numSV]*SigRecon*VT[:numSV,:]
    print("*****reconstruct matrix using %d singular values*****" %numSV)
    printMat(reconMat,thresh)

#myMat=mat(loadExData())
#myMat[0,1]=myMat[0,0]=myMat[1,0]=myMat[2,0]=4
#myMat[3,3]=2
#result1=recommend(myMat,2,simMeas=ecludSim)
#result2=recommend(myMat,2,simMeas=pearsSim)
#result3=recommend(myMat,1,estMethod=svdEst)
#result4=recommend(myMat,1,estMethod=svdEst,simMeas=pearsSim)
result=imagePress(2)
