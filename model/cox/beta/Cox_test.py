import pandas as pd
import statsmodels.formula.api as smf
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.externals import joblib
import matplotlib.pylab as plt


record_keys = []


def initial(file, x, y, id):
    data = pd.read_csv(file)
    # data = data.dropna()

    data['L0005'] = data['L0005'] - 10600
    # 1010 1031 1032 1036 2010 2020
    # data['L0020'] = data['L0020'].map({1010: 0, 1031: 1, 1032: 2, 1036: 3, 2010: 4, 2020: 5})
    # print(data['L0020'])
    print(data.head())
    print("最开始的数量：", len(data))

    # 切分为训练集和测试集
    titleList = data.columns.values.tolist()
    print(titleList)
    x_keys = []
    for key in x:
        x_keys.append(key)
    x_keys.append(y)
    x_keys.append(id)
    # y_keys = []
    for a in titleList:
        if a not in x_keys:
            del data[a]
    print(data.head())
    testData = data[x_keys]
    return testData




# 将data数据存入list record中
# 构建存活时间与其他值的键值对
def getRecord(data, y, id):
    record = {}
    for index, row in data.iterrows():
        oneRecord = {}
        futime = row[y]
        for x in record_keys:
            oneRecord[x] = row[x]
        oneRecord[id] = row[id]
        record.setdefault(futime, []).append(oneRecord)
    num = 0
    for time in record:
        num = num + len(record[time])
    print("记录条数：", num)
    return record

# 测试，得到预测准确率
def predict(record, params, S0, id):
    IDandSList = []
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
                if x is not id:
                    temp = temp + value[x] * params[x]
            b = math.exp(temp)
            S = math.pow(S0Test, b)
            IDandSList.append((value[id], 1-S))

    with open('result', 'a') as f:
        for a, b in IDandSList:
            aSplit = str(a).split('.')
            f.write(aSplit[0])
            f.write(',')
            f.write(str(b))
            f.write('\n')



# initial(file) return trainData,testData,params
# getRecord(data, y, state) return record
# getS0(record, params) return S0
# getEvaluation(record, testTime) return evaluation
# predict(record, params, S0, evaluation, testTime)
if __name__ == '__main__':
    file = "D:\TelecomData\\test.csv"

    x = ["L0005", "L0023", "L0016", "L0020", "L0074", "L0014", "L0117", "L0114", "L0189", "L0215", "L0256", "L0061",
         "L0072", "L0134"]

    record_keys = x
    y = 'L0009'
    id = "L0001"

    model = joblib.load('Cox.model')

    testData = initial(file, x, y, id)
    # trainRecord = getRecord(trainData, y, state)
    testRecord = getRecord(testData, y, id)

    params = model["params"]
    S0_t = model["S0"]
    predict(testRecord, params, S0_t, id)

