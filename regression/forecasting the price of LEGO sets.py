from time import sleep
import json
import urllib.request
from bs4 import BeautifulSoup
from numpy import *
import numpy as np

'''
def searchForSet(retX,retY,setNum,yr,numPce,origPrc):
    sleep(10)
    myAPIstr='AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY'
    searchURL='https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json'\
              %(myAPIstr,setNum)
    pg=urllib.request.urlopen(searchURL)
    retDict=json.loads(pg.read())
    for i in range(len(retDict['items'])):
        try:
            currItem=retDict['items'][i]
            if currItem['product']['condition']=='new':
                newFlag=1
            else:
                newFlag=0
            listOfInv=currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice=item['price']
                if sellingPrice>origPrc*0.5:
                    print("%d\t%d\t%d\t%f\t%f" %\
                          (yr,numPce,newFlag,origPrc,sellingPrice))
                    retX.append([yr,numPce,newFlag,origPrc])
                    retY.append(sellingPrice)
        except:
            print("problem with item %d" %i)'''

def scrapgePage(inFile,outFile,yr,numPce,origPrc):
    fr=open(inFile,encoding='utf-8'); fw=open(outFile,'a')
    soup=BeautifulSoup(fr.read())
    i=1
    currentRow=soup.findAll('table',r="%d" %i)
    while(len(currentRow)!=0):
        title=currentRow[0].findAll('a')[1].text
        lwrTitle=title.lower()
        if (lwrTitle.find('new')>-1) or (lwrTitle.find('nisb')>-1):
            newFlag=1.0
        else:
            newFlag=0.0
        soldUnicde=currentRow[0].findAll('td')[3].findAll('span')
        if len(soldUnicde)==0:
            print("item #%d did not sell" %i)
        else:
            soldPrice=currentRow[0].findAll('td')[4]
            priceStr=soldPrice.text
            priceStr=priceStr.replace('$','')#strips out $
            priceStr=priceStr.replace(',','')#strips out ,
            if len(soldPrice)>1:
                priceStr=priceStr.replace('Free shipping','')
            print("%s\t%d\t%s" %(priceStr,newFlag,title))
            fw.write("%d\t%d\t%d\t%f\t%s\n" %(yr,numPce,newFlag,origPrc,priceStr))
        i+=1
        currentRow=soup.findAll('table',r="%d" %i)
    fw.close()


'''
def setDataCollect(retX,retY):
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)'''

def setDataCollect(outFile):
    scrapgePage('setHtml/lego8288.html',outFile,2006,800,49.99)
    scrapgePage('setHtml/lego10030.html',outFile,2002,3096,269.99)
    scrapgePage('setHtml/lego10179.html', outFile, 2007, 5195, 499.99)
    scrapgePage('setHtml/lego10181.html', outFile, 2007, 3428, 199.99)
    scrapgePage('setHtml/lego10189.html', outFile, 2008, 5922, 299.99)
    scrapgePage('setHtml/lego10196.html', outFile, 2009, 3263, 249.99)

def loadDataSet(fileName):
    numFeat=len(open(fileName).readline().split('\t'))-1
    xMat=[];yMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        xMat.append(lineArr)
        yMat.append(float(curLine[-1]))
    return xMat,yMat

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

def crossValidation(xArr,yArr,numVal=10):
    m=len(yArr)
    indexList=list(range(m))
    errorMat=zeros((numVal,30))
    for i in range(numVal):
        trainX=[];trainY=[]
        testX=[];testY=[]
        random.shuffle(indexList)
        #90%训练+10%测试
        for j in range(m):
            if j<m*0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat=ridgeTest(trainX,trainY)
        for k in range(30):
            matTestX=mat(testX);matTrainX=mat(trainX)
            #用训练集参数将测试集数据标准化
            meanTrain=mean(matTestX,0)
            varTrain=var(matTrainX,0)
            matTestX=(matTestX-meanTrain)/varTrain
            yEst=matTestX*mat(wMat[k,:]).T+mean(trainY)
            errorMat[i,k]=rssError(yEst.T.A,array(testY))
            
    #计算不同岭回归ws下errorMat的平均值，观察平均性能
    meansErrors=mean(errorMat,0)
    minMean=float(min(meansErrors))
    bestWeights=wMat[nonzero(meansErrors==minMean)]

    # 岭回归使用了数据标准化，而standRegres没有，为了比较可视化，因此需要将数据还原
    # 标准化后 Xreg = (x-meanX)/var(x)，预测y=Xreg*w+meanY
    # 因此，利用未标准化的x来计算y= x*w/var(x) - meanX*w/var(x) +meanY
    # 其中unReg=w/var
    xMat=mat(xArr);yMat=mat(yArr).T
    meanX=mean(xMat,0);varX=var(xMat,0)
    unReg=bestWeights/varX
    #特别注意这里的sum函数，一定是np.sum,因为一般的sum只能对list求和，而这里的参数是matrix
    yHat=xMat*unReg.T-1*np.sum(multiply(meanX,unReg))+mean(yMat)
    return yHat


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

def rssError(yArr,yHat):
    return ((yArr-yHat)**2).sum()

if __name__=='__main__':
    setDataCollect('result,txt')
    xMat,yMat=loadDataSet('result.txt')
    lgx=mat(ones((1261,5)))
    lgx[:,1:5]=mat(xMat)
    lgY=mat(yMat).T
    ws=standRegres(lgx,mat(yMat))
    yHat=lgx*ws
    err1=rssError(lgY.A,yHat.A)
    cor1=corrcoef(yHat.T,lgY.T)
    yHat2=crossValidation(xMat,yMat,numVal=10)
    err2=rssError(lgY.A,yHat2.A)
    cor2=corrcoef(yHat2.T,lgY.T)