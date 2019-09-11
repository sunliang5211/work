import pandas as pd
import statsmodels.formula.api as smf
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.externals import joblib
import matplotlib.pylab as plt


record_keys = []


def initial(trainFile, testFile, x, y, state):
    data1 = pd.read_csv(trainFile)
    data2 = pd.read_csv(testFile)

    data1['L0005'] = data1['L0005'] - 10600
    data2['L0005'] = data2['L0005'] - 10600
    print(data2.head())
    print("训练集数量：", len(data1))
    print("测试集数量：", len(data2))

    titleList = data1.columns.values.tolist()
    print(titleList)
    x_keys = []
    for key in x:
        x_keys.append(key)
    x_keys.append(y)
    x_keys.append(state)
    for a in titleList:
        if a not in x_keys:
            del data1[a]
            del data2[a]
    print(data1.head())
    print(data2.head())
    trainData = data1[x_keys]
    testData = data2[x_keys]

    status = trainData[state].values

    sentence = y + "~"
    count = 0
    for key in record_keys:
        if count == 0:
            sentence = sentence + key
        else:
            sentence = sentence + "+" + key
        count = count + 1
    print(sentence)

    # mod = smf.phreg("futime ~ age + female + creatinine + "
    #                 "  + year",
    #                 trainData, status=status, ties="efron")

    mod = smf.phreg(sentence, trainData, status=status, ties="efron")
    rslt = mod.fit()
    print(rslt.summary())
    # 得到h(t|X)=h0(t)exp(X^T*B)的协变量参数B
    params = {}
    i = 0
    while i < len(record_keys):
        params[record_keys[i]] = rslt.params[i]
        i = i + 1
    print(params)
    return trainData, testData, params




# 将data数据存入list record中
# 构建存活时间与其他值的键值对
def getRecord(data, y, state):
    record = {}
    for index, row in data.iterrows():
        oneRecord = {}
        futime = row[y]
        death = row[state]
        for x in record_keys:
            oneRecord[x] = row[x]
        oneRecord[state] = death
        record.setdefault(futime, []).append(oneRecord)
    num = 0
    for time in record:
        num = num + len(record[time])
    print("记录条数：", num)
    return record

# 用Breslow法估计出基准生存函数S0(ti)
# h0为基准风险函数，H0为基准累积风险函数，S0为基准生存率
def getS0(record, params, state):
    h0 = {}
    H0 = {}

    for time in record:
        a = len(record[time])
        for value in record[time]:
            if int(value[state]) is 0:
                a = a - 1
        sumb = 0
        for time2 in record:
            if time <= time2:
                for value in record[time2]:
                    temp = 0
                    for x in value:
                        if x is not state:
                            temp = temp + value[x] * params[x]
                    b = math.exp(temp)
                    sumb = sumb + b
        h0[time] = a / sumb
    print(h0)
    for time in h0:
        temp = 0
        for time2 in h0:
            if time2 < time:
                temp = temp + h0[time2]
        H0[time] = temp
    print(H0)

    # 得到S0
    S0 = {}
    for time in H0:
        S0[time] = math.exp(H0[time] * (-1))
    sort_S0 = sorted(S0.items(), key=lambda x: x[0])
    S0 = dict(sort_S0)
    print(S0)

    timeList = []
    S0List = []
    for key, value in S0.items():
        timeList.append(key)
        S0List.append(value)

    x = np.array(timeList)
    y = np.array(S0List)

    f1 = np.polyfit(x, y, 2)
    S0_t = np.poly1d(f1)
    print("p1 is :", S0_t)

    return S0_t

# 测试，得到预测准确率/召回率/F1
def predict(record, params, state, S0):

    deathNum = 0
    censorNum = 0
    for time in record:
        for value in record[time]:
            if value[state] == 0:
                censorNum = censorNum + 1
            else:
                deathNum = deathNum + 1
    print("测试集中删失数据数量为：", censorNum)
    print("测试集中流失数据数量为，", deathNum)

    SandStateList = []

    for time in record:
        # matchtime = 0
        # for time2 in S0:
        #     if matchtime < time2 <= time:
        #         matchtime = time2
        # S0Test = S0[matchtime]

        S0Test = S0(time)
        if S0Test < 0:
            S0Test = 0
        # print(str(time) + "," + str(S0Test))

        for value in record[time]:
            temp = 0
            for x in value:
                if x is not state:
                    temp = temp + value[x] * params[x]
            b = math.exp(temp)
            S = math.pow(S0Test, b)
            SandStateList.append((1-S, value[state]))

    SandStateList.sort(reverse=True)
    print(SandStateList)

    count = 0
    match = 0
    while count < 1000:
        if SandStateList[count][1] == 1:
            match += 1
        count += 1
    print("----------------------------------------")
    print("此时取前1000个离网概率最大的用户")
    print("match:", match)
    precision = match/1000
    print("准确率为:", precision)
    recall = match/1000
    print("召回率为：", recall)
    F1 = 2*precision*recall/(precision+recall)
    print("F1为：", F1)

    for a in [5, 6, 7, 8, 9]:
        print("----------------------------------------")
        print("此时取阈值为" + str(a) +"0%")
        count = 0
        match = 0
        s = 1
        b = a/10
        while s >= b:
            if SandStateList[count][1] == 1:
                match += 1
            count += 1
            s = SandStateList[count][0]
        print("离网概率大于"+ str(a) +"0%的有：", count)
        print("并且是真实离网的有：", match)
        precision = match / count
        print("准确率为:", precision)
        recall = match / 1000
        print("召回率为：", recall)
        F1 = 2 * precision * recall / (precision + recall)
        print("F1为：", F1)


# initial(file) return trainData,testData,params
# getRecord(data, y, state) return record
# getS0(record, params) return S0
# getEvaluation(record, testTime) return evaluation
# predict(record, params, S0, evaluation, testTime)
if __name__ == '__main__':
    # half&half2
    trainFile = "D:\data\sun\survival\\train.csv"
    testFile = "D:\data\sun\survival\\test.csv"

    # 第一次
    x = ["L0005", "L0023", "L0016", "L0020", "L0074", "L0014", "L0117", "L0114", "L0189", "L0215", "L0256", "L0061",
         "L0072", "L0134"]

    record_keys = x
    y = 'L0009'
    state = 'death'
    trainData, testData, params = initial(trainFile, testFile, x, y, state)
    trainRecord = getRecord(trainData, y, state)
    testRecord = getRecord(testData, y, state)
    S0_t = getS0(trainRecord, params, state)

    # 保存模型
    model = {"params": params, "S0": S0_t}
    joblib.dump(model, 'D:\\work\\model\\cox\\beta\\Cox.model')

    predict(testRecord, params, state, S0_t)

