# -*-coding:utf-8 -*-
import operator
import feedparser
import random
import re
from numpy import *

def textParse(bigString):
    '''
    Takes a big String and parses out the text into a list of strings
    element anything under two characters long and convert everything into lowercase.
    '''
    listOfTokens=re.split(r'\W*',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok)>2]

def createVocabList(dataSet):
    vocabSet=set([])
    for document in dataSet:
        vocabSet=vocabSet | set(document)
    return list(vocabSet)

def bagOfWords2VecMN(vocabList,inputSet):
    '''
    change to vector
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
    p0Num=ones(numWords)
    p1Num=ones(numWords)
    p0Denom=2.0
    p1Denom=2.0
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            p1Num+=trainMartix[i]# increment that token
            p1Denom+=sum(trainMartix[i])# increment the tokens
        else:
            p0Num+=trainMartix[i]
            p0Denom+=sum(trainMartix[i])
    #begin with p1Num/p1Denom will have under flow
    #change to log avoid this mistake.
    p1Vect=log(p1Num/p1Denom)
    p0Vect=log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2classify,p0Vec,p1Vec,pClass1):
    p1=sum(vec2classify*p1Vec)+log(pClass1)
    p0=sum(vec2classify*p0Vec)+log(1.0-pClass1)
    if p1>p0:
        return 1
    else:
        return 0


def calcMostFreq(vocabList,fullText):
    '''
    go through every word in the vocabulary and counts
    how many time it appears in the text.
    '''
    freDict={}
    for token in vocabList:
        freDict[token]=fullText.count(token)
    #In python 3.x iteritems() change into items()
    sortedFreq=sorted(freDict.items(),key=operator.itemgetter(1),reverse=True)
    return sortedFreq[:30]

def localWords(feed1,feed0):
    docList=[]
    classList=[]
    fullText=[]
    minLen=min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList=textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)

        wordList=textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    vocabList=createVocabList(docList)
    top30Words=calcMostFreq(vocabList,fullText)

    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])

    trainingSet=range(2*minLen)
    testSet=[]

    for i in range(20):
        randIndex=int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])

    trainMat=[]
    trainClasses=[]

    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])

    p0V,p1V,pSpam=trainNB0(array(trainMat),array(trainClasses))
    errorcount=0

    for docIndex in testSet:
        wordVector=bagOfWords2VecMN(vocabList,docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam)!=classList[docIndex]:
            errorcount+=1
    print('The error rate is:',float(errorcount)/len(testSet))
    return vocabList,p0V,p1V

def getTopWords(ny,sf):
    vocabList,p0V,p1V=localWords(ny,sf)
    topNY=[]
    topSF=[]
    for i in range(len(p0V)):
        if p0V[i]>-6.0:topSF.append((vocabList[i],p0V[i]))
        if p1V[i]>-6.0:topNY.append((vocabList[i],p1V[i]))
    soretedSF=sorted(topSF,key=lambda pair:pair[i],reverse=True)
    print("SF*SF*SF*SF*SF*SF*SF*SF*SF*SF*SF*SF*SF*SF*SF*SF*SF*SF*SF*")
    for item in soretedSF:
        print(item[0])
    sortedNY=sorted(topNY,key=lambda pair:pair[1],reverse=True)
    print("NY*NY*NY*NY*NY*NY*NY*NY*NY*NY*NY*NY*NY*NY*NY*NY*NY*NY*NY*")
    for item in sortedNY:
        print(item[0])

if __name__=='__main__':
    ny=feedparser.parse('http//newyork.craigslist.org/stp/index.rss')
    sf=feedparser.parse('http//sfbay.craigslist.org/stp/index.rss')
    vocablist,pSF,pNY=localWords(ny,sf)
    #vocablist,pSF,pNY=localWords(ny,sf)
    getTopWords(ny,sf)
    #IndexError: range object index out of range: reason: the source of data have some problem.
