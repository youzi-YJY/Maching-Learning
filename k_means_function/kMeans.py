from numpy import *
import matplotlib.pyplot as plt
import json
import urllib.request
from urllib.parse import urlencode
from time import sleep

def loadDataSet(filename):
    dataMat=[]
    fr=open(filename)
    for line in fr.readlines():
        curLine=line.strip().split('\t')
        fltLine=list(map(float,curLine))
        dataMat.append(fltLine)
    return dataMat

def distEclud(vecA,vecB):
    return sqrt(sum(power(vecA-vecB,2)))

def randCent(dataSet,k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))
    for j in range(n):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)
    return centroids

def kMeans(dataSet,k,disMeas=distEclud,createCent=randCent):
    m=shape(dataSet)[0]
    #clusterAssment has to columns,one column is for the index of cluster and second column is to store
    #the error,this error is the distance from the cluster centroid to the current point.
    clusterAssment=mat(zeros((m,2)))
    centroids=createCent(dataSet,k)
    clusterChanged=True
    while clusterChanged:
        clusterChanged=False#如果没有更新则为退出
        for i in range(m):
            minDist=inf
            minIndex=-1
            for j in range(k):#每个样本点需要与所有的中心点作比较
                distJI=disMeas(centroids[j,:],dataSet[i,:])#距离计算
                if distJI<minDist:
                    minDist=distJI
                    minIndex=j
            if clusterAssment[i,0]!=minIndex:#若记录矩阵的i样本的所属中心点更新，则为True，while下次继续循环更新
                clusterChanged=True
            clusterAssment[i,:]=minIndex,minDist**2#记录该点的两个信息
        #print(centroids)
        for cent in range(k):#重新计算中心点
            ptsInClust=dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]#得到属于该中心点的所有样本数据
            centroids[cent,:]=mean(ptsInClust,axis=0)#求每列的均值替换原来的中心点
    #the centroids and cluster assignments are returned
    return centroids,clusterAssment

def biKmeans(dataSet,k,distMeas=distEclud):
    m=shape(dataSet)[0]
    clusterAssment=mat(zeros((m,2)))#保存数据点的信息：类别和误差
    centroid0=mean(dataSet,axis=0).tolist()[0]#根据数据集均值获得第一个簇中心点
    centList=[centroid0]#创建一个带有质心的列表，后面会加入k个质心
    for j in range(m):
        clusterAssment[j,1]=distMeas(mat(centroid0),dataSet[j,:])**2#求得dataSet点与质心点的SSE
    while (len(centList)<k):
        lowestSSE=inf
        for i in range(len(centList)):
            ptsInCurrCluster=dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]#获得属于该质心点的所有样本数据
            #二分类
            centroidMat,splitClustAss=kMeans(ptsInCurrCluster,2,distMeas)#返回中心点信息、该数据聚类信息
            sseSplit=sum(splitClustAss[:,1])#划分数据的SSE 加上未划分的 作为本次划分的总误差
            sseNotSplit=sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])#未划分数据集的SSE
            print("sseSplit,and notSplit:",sseSplit,sseNotSplit)
            if (sseSplit+sseNotSplit)<lowestSSE:#将划分与未划分的SSE求和与最小的SSE相比较 确定是否划分
                bestCenToSplit=i #划分中心点
                bestNewCents=centroidMat #划分后的两个新中心点
                bestClusterAss=splitClustAss.copy() #划分点的聚类信息
                lowestSSE=sseSplit+sseNotSplit
        bestClusterAss[nonzero(bestClusterAss[:,0].A==1)[0],0]=len(centList) #将属于1的所属信息转为下一个中心点
        bestClusterAss[nonzero(bestClusterAss[:,0].A ==0)[0],0]=bestCenToSplit #将属于0的所属信息替换用来聚类的中心点
        print('the bestCentToSplit is(本次最适合划分的质心点) :',bestCenToSplit)
        print('the len of bestClustAss is(被划分数据数量):',len(bestClusterAss))
        centList[bestCenToSplit]=bestNewCents[0,:].tolist()[0]#替换中心点信息，上面是替换数据点所属信息。
        centList.append(bestNewCents[1,:].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:,0].A==bestCenToSplit)[0],:]=bestClusterAss#替换部分用来聚类的数据的所属中心点和误差平方和为新的数据
    return mat(centList),clusterAssment

def plot_show(cenList,dataMat):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(cenList[:, 0].flatten().A[0], cenList[:, 1].flatten().A[0], color='r', s=60)
    ax.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0])
    plt.show()

#以上测试过，没有问题.

def geoGrab(stAddress,city):
    apiStem='http://where.yahooapis.com/geocode?'
    params={}
    params['flags']='J'
    params['appid']='ppp68N8t'
    params['location']='%s %s' %(stAddress,city)
    url_params=urlencode(params)
    yahooApi=apiStem+url_params
    print(yahooApi)
    c=urllib.request.urlopen(yahooApi)
    return json.loads(c.read())

def massPlaceFind(fileName):
    fw=open('data/places.txt','w')
    for line in open(fileName).readlines():
        line=line.strip()
        lineArr=line.split('\t')
        retDict=geoGrab(lineArr[1],lineArr[2])
        if retDict['ResultSet']['Error']==0:
            lat=float(retDict['ResultSet']['Results'][0]['latitude'])
            lng=float(retDict['ResultSet']['Results'][0]['longitude'])
            print("%s\t%f\t%f" %(lineArr[0],lat,lng))
            fw.writelines('%s\t%f\t%f\n' %(line,lat,lng))
        else:
            print("error fetching")
        sleep(1)
    fw.close()

def distSLC(vecA,vecB):
    a=sin(vecA[0,1]*pi/180)*sin(vecB[0,1]*pi/180)
    b=cos(vecA[0,1]*pi/180)*cos(vecB[0,1]*pi/180)*cos(pi*(vecB[0,0]-vecA[0,0])/180)
    return arccos(a+b)*6371.0

def clusterClubs(numClust=5):
    datList = []
    for line in open('data/places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    fig = plt.figure()
    rect=[0.1,0.1,0.8,0.8]
    scatterMarkers=['s', 'o', '^', '8', 'p', \
                    'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0=fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1=fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A==i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)
    ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)
    #plt.show()
    plt.savefig('result.png',bbox_inches='tight')

if __name__=='__main__':
    data=loadDataSet('data/testSet.txt')
    dataM=mat(data)
    dataMat,clusterAssment=biKmeans(dataM,2)
    plot_show(dataMat,dataM)