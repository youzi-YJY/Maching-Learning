from time import sleep
from votesmart import votesmart
from numpy import *
votesmart.apikey='49024thereoncewasamanfromnantucket94040'


def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])

    C1.sort()
    return list(map(frozenset, C1))  # use frozen set so we
    # can use it as a key in a dict


def scanD(D, Ck, minSupport):
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if can not in ssCnt.keys() :
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key] / numItems
        if support >= minSupport:
            retList.insert(0, key)
            supportData[key] = support
    return retList, supportData

def aprioriGen(Lk,k):#create Ck
    '''
    生成所有可以组合的集合
    频繁项集列表Lk,项集元素个数K.
    '''
    retList=[]
    lenLk=len(Lk)
    #join sets if first k-2 items are equal
    for i in range(lenLk):#两层循环比较Lk中的每个元素与其它元素
        for j in range(i+1,lenLk):
            L1=list(Lk[i])[:k-2];L2=list(Lk[j])[:k-2]
            L1.sort();L2.sort()
            if L1==L2:
                retList.append(Lk[i] | Lk[j])#set union which is the | symbol in Python.
    return retList

def apriori(dataSet,minSupport=0.5):
    '''
    封装所有步骤的函数
    返回所有满足大于阈值的组合 集合支持度列表
    '''
    C1=createC1(dataSet)
    D=list(map(set,dataSet))
    L1,supportData=scanD(D,C1,minSupport)#过滤数据
    L=[L1]
    k=2
    while(len(L[k-2])>0):#若仍有满足支持度的集合规则则继续做关联分析
        Ck=aprioriGen(L[k-2],k)#Ck候选频繁项集
        Lk,supK=scanD(D,Ck,minSupport)#Lk频繁项集
        supportData.update(supK)#更新字典(把新出现的集合：支持度加入到supportData中)
        L.append(Lk)
        k+=1#每次新组合的元素都只增加了一个，所有k也加一(k代表元素个数)
    return L,supportData


# 对规则进行评估 获得满足最小可信度的关联规则
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = []  # 创建一个新的列表去返回
    for conseq in H:
        conf = supportData[freqSet] / supportData[freqSet - conseq]  # 计算置信度
        if conf >= minConf:
            print(freqSet - conseq, '-->', conseq, 'conf:', conf)
            brl.append((freqSet - conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

# 生成候选规则集合
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    if (len(freqSet) > (m + 1)):  # 尝试进一步合并
        Hmp1 = aprioriGen(H, m + 1)  # 将单个集合元素两两合并
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if (len(Hmp1) > 1):  # need at least two sets to merge
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)

# 获取关联规则的封装函数
def generateRules(L, supportData, minConf=0.7):  # supportData 是一个字典
    bigRuleList = []
    for i in range(1, len(L)):  # 从为2个元素的集合开始
        for freqSet in L[i]:
            # 只包含单个元素的集合列表
            H1 = [frozenset([item]) for item in freqSet] # frozenset({2, 3}) 转换为 [frozenset({2}), frozenset({3})]
            # 如果集合元素大于2个，则需要处理才能获得规则
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)  # 集合元素 集合拆分后的列表 。。。
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList

#Error
def getActionIds():
    actionIdList=[]
    billTitleList=[]
    fr=open('/Users/yangjiayuan/PycharmProjects/Machine Learning/association rule/data/recent20bills.txt')
    for line in fr.readlines():
        billNum=int(line.split('\t')[0])
        try:
            billDetail=votesmart.votes.getBill(billNum)
            for action in billDetail.actions:
                if action.level=='House' and (action.stage=='Passage' or action.stage=='Amendment Vote'):
                    actionId=int(action.actionId)
                    print('bill: %d has actionId: %d' %(billNum,actionId))
                    actionIdList.append(actionId)
                    billTitleList.append(line.strip().split('\t')[1])
        except:
            print("problem getting bill %d" %billNum)
        sleep(1)
    return actionIdList,billTitleList


def getTrainsList(actionIdList,billTitleList):
    itemMeaning=['Republican','Democractic']
    for billTitle in billTitleList:
        itemMeaning.append('%s --Nay' %billTitle)
        itemMeaning.append('%s --Yea' %billTitle)
    transDict={}
    voteCount=2
    for actionId in actionIdList:
        sleep(3)
        print('getting votes for action: %d' %actionId)
    try:
        voteList=votesmart.votes.getBillActionVotes(actionId)
        for vote in voteList:
            if not vote.candidateName in transDict:
                transDict[vote.candidateName]=[]
                if vote.officeParties=='Democratic':
                    transDict[vote.candidateName].append(1)
                elif vote.officeParties=='Republican':
                    transDict[vote.candidateName].append(0)
            if vote.action=='Nay':
                transDict[vote.candidateName].append(voteCount)
            elif vote.action=='Yea':
                transDict[vote.candidateName].append(voteCount+1)
    except:
        print("problem getting actionId: %d" % actionId)
    voteCount+=2
    return transDict,itemMeaning

#以上测试过，没有问题。
if __name__=='__main__':
    dataSet=loadDataSet()
    #C1=createC1(dataSet)
    #D=list(map(set,dataSet))
    #L1,suppData0=scanD(D,C1,0.5)
    L,suppData=apriori(dataSet,minSupport=0.5)
    #rules=generateRules(L,suppData0,minConf=0.5)
    rules=generateRules(L,suppData,minConf=0.5)
