# -*- coding:utf-8 -*-

from numpy import *

def loadDataset():
    '''
    create some example data to experiment with.
    Word list to vector function
    :return:
    '''
    #Row data
    postingList=[
        ['my','dog','has','flea','problems','help','please'],
        ['maybe','not','take','him','to','dog','park','stupid'],
        ['my','dalmation','is','so','cute','I','love','him'],
        ['stop','posting','stupid','worthless','garbage'],
        ['mr','licks','ate','my','steak','how','to','stop','him'],
        ['quit','buying','worthless','dog','food','stupid']
    ]
    #Here we have two classes, abusive and not abusive.
    classVec=[0,1,0,1,0,1]# 1 is abusive, 0 not.
    return postingList,classVec

def createVocabList(dataSet):
    vocabSet=set([])
    for document in dataSet:
        vocabSet=vocabSet | set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList,inputSet):
    '''
    takes the vocabulary list and a document and outputs a vector
    of 1s and 0s to represent whether a word from our vocabulary
    is present or not in the given document.every time it encounters
    a word, it increments the word vector rather than setting the word
     vector to 1 for a given index.
    '''
    returnVec=[0]*len(vocabList)#create a  vector of all 0s
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]+=1
    return returnVec


def trainNB0(trainMartix,trainCategory):
    '''
    Naive Bayes classifier training function
    :param trainMartix: a matrix of documents
    :param trainCategory: a vector with the class labels for each of documents.
    :return:
    '''
    numTrainDocs=len(trainMartix)#trainMartix数组总长度
    numWords=len(trainMartix[0])#trainMartix中单个数组的长度
    #calculate the probability the document is an abusive document
    pAbusive=sum(trainCategory)/float(numTrainDocs)#trainCategory在numTrainDocs里面占了多少
    #begin with zeros and 0.0
    #if we multiple every result, where exist 0, we will get the 0
    #This make no sense, make a change.
    p0Num=zeros(numWords)
    p1Num=zeros(numWords)
    p0Denom=0.0
    p1Denom=0.0
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            p1Num+=trainMartix[i]# increment that token
            p1Denom+=sum(trainMartix[i])# increment the tokens
        else:
            p0Num+=trainMartix[i]
            p0Denom+=sum(trainMartix[i])
    #begin with p1Num/p1Denom will have underflow
    #change to log avoid this mistake.
    p1Vect=p1Num/p1Denom
    p0Vect=p0Num/p0Denom # if change to log,the initial martix must be ones. Raise RuntimeWarning:divide by zero encountered in log.
    return p0Vect,p1Vect,pAbusive

def classifyNB(vecClassify,p0Vec,p1Vec,pClass1):
    p1=sum(vecClassify*p1Vec)+log(pClass1)
    p0=sum(vecClassify*p0Vec)+log(1.0-pClass1)
    if p1>p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts,listClasses=loadDataset()
    myVocabList=createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb=trainNB0(array(trainMat),array(listClasses))
    testEntry1=['love','my','dalmation']
    thisDoc1=array(setOfWords2Vec(myVocabList,testEntry1))
    print(testEntry1,'classified as:',classifyNB(thisDoc1,p0V,p1V,pAb))
    testEntry2=['stupid','garbage']
    thisDoc2=array(setOfWords2Vec(myVocabList,testEntry2))
    print(testEntry2, 'classified as:', classifyNB(thisDoc2, p0V, p1V, pAb))

if __name__=='__main__':
    testingNB()