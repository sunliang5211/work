import pandas as pd
import math
from sklearn.externals import joblib
import os


record_keys = []


def initial(data, titleList, keys):
    # data = pd.read_csv(file)
    # data = data.dropna()

    # 只取相关列数
    for a in titleList:
        if a not in keys:
            del data[a]

    # 把"-"替换为"-1"，把空值赋值为0
    # titleList2 = data.columns.values.tolist()
    for a in keys:
        data.ix[data[a] == "-", a] = "-1"
        data[a] = data[a].fillna(0)

    data['L0005'] = data['L0005'] - 10600
    # 1010 1031 1032 1036 2010 2020
    # data['L0020'] = data['L0020'].map({1010: 0, 1031: 1, 1032: 2, 1036: 3, 2010: 4, 2020: 5})
    # print(data['L0020'])
    print(data.head())


    return data


def dataPreprocessing(file):
    with open(file, 'rb') as r:
        lines = r.readlines()
    with open(file, 'w') as w:
        for line in lines:
            try:
                aSplit = line.decode('utf8').split('\n')
                w.write(aSplit[0])
                # 为了暴露出错误，最好此处不print
            except:
                print(str(line))




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
    file = "D:\TelecomData\\zaiwang_user.csv"
    outputFile = "result"

    # 如果输出文件存在则删除
    if os.path.exists(outputFile):
        os.remove(outputFile)

    # 数据预处理，第一次运行这份数据时需要调用此函数
    dataPreprocessing(file)


    x = ["L0005", "L0023", "L0016", "L0020", "L0074", "L0014", "L0117", "L0114", "L0189", "L0215", "L0256", "L0061",
         "L0072", "L0134"]

    record_keys = x
    y = 'L0009'
    id = "L0001"
    keys = []
    for key in x:
        keys.append(key)
    keys.append(y)
    keys.append(id)


    model = joblib.load('Cox.model')
    params = model["params"]
    S0_t = model["S0"]
    print("params:", params)
    print("S0:", S0_t)


    flag = 0

    reader = pd.read_csv(file, sep=',', chunksize=100000)
    for chunk in reader:
        if flag == 0:
            titleList = chunk.columns.values.tolist()
            print(titleList)
            flag = 1

        testData = initial(chunk, titleList, keys)
        # trainRecord = getRecord(trainData, y, state)
        testRecord = getRecord(testData, y, id)

        predict(testRecord, params, S0_t, id)







