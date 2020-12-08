import re
import random
from numpy import *

def createVocabList(dataSet):
    vocabSet=set([])
    for document in dataSet:
        vocabSet=vocabSet | set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList,inputSet):
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

def textParse(bigString):
    '''
    Takes a big String and parses out the text into a list of strings
    element anything under two characters long and convert everything into lowercase.
    '''
    listOfTokens=re.split(r'\W*',bigString)#非单词字符 0次或无数次
    return [tok.lower() for tok in listOfTokens if len(tok)>2]

def spamTest():
    docList=[]
    classList=[]
    fullText=[]
    for i in range(1,26):
        #UnicodeDecodeError:'utf8'codec can`t decode byte Ox9c, solution:add errors='ignore'.
        wordList=textParse(open('email/spam/%d.txt' %i,errors='ignore').read())
        #Attention the differnece between append and extend.
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)

        wordList=textParse(open('email/ham/%d.txt' %i,errors='ignore').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    vocabList=createVocabList(docList)
    #TypeError:'range' object doesn`t support item deletion,solution: change range(50) into list(range(50))
    trainingSet=list(range(50))
    testSet=[]

    for i in range(10):
        #randomly select 10 items from trainingSet as testSet every time.
        #give a number,range in (0,len(trainingSet)
        randIndex=int(random.uniform(0,len(trainingSet)))
        #As a number is selected, it`s added to the test set and removed from the trainingSet.
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])

    trainMat=[]
    trainClasses=[]

    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam=trainNB0(array(trainMat),array(trainClasses))
    errcount=0

    for docIndex in testSet:
        wordVector=setOfWords2Vec(vocabList,docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam)!=classList[docIndex]:
            errcount+=1
    print('The error rate is:',float(errcount)/len(testSet))

if __name__=='__main__':
    spamTest()