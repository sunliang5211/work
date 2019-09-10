import pandas as pd


# dataZaiwang = pd.read_csv("D:\TelecomData\zaiwang_user.csv", sep=',', engine='python', iterator=True)
dataZaiwang = pd.read_csv("D:\TelecomData\zaiwang_user.csv", sep=',',engine = 'python',iterator=True)
dataLiwang = pd.read_csv("D:\TelecomData\liwang_user.csv")

lenOfTrain = int(len(dataLiwang)*0.6)
chunkSize = lenOfTrain + 1000 * 100
dataZaiwang = dataZaiwang.get_chunk(chunkSize)

dataZaiwang['death'] = 0
dataLiwang['death'] = 1

data_merge = pd.concat([dataLiwang, dataZaiwang])
print(len(data_merge))


# 删除空列
emptyList = ["L0004", "L0045", "L0046", "L0047", "L0048", "L0056", "L0057", "L0058", "L0059", "L0060", "L0065", "L0066",
             "L0067", "L0068"]
for a in emptyList:
    del data_merge[a]

# 把"-"替换为"-1"，把空值赋值为0
titleList = data_merge.columns.values.tolist()
for a in titleList:
    data_merge.ix[data_merge[a] == "-", a] = "-1"
    data_merge[a] = data_merge[a].fillna(0)

# #显示所有列
pd.set_option('display.max_columns', None)

print(len(data_merge))
print(data_merge.head())


dataLiwangTrain = data_merge.loc[data_merge['death'] == 1].head(lenOfTrain)
plus1000 = lenOfTrain+1000
dataLiwangTest = data_merge.loc[data_merge['death'] == 1][lenOfTrain:plus1000]
print(dataLiwangTest)


dataZaiwangTrain = data_merge.loc[data_merge['death'] == 0].head(lenOfTrain)
dataZaiwangTest = data_merge.loc[data_merge['death'] == 0][lenOfTrain:]

dataTrain = pd.concat([dataLiwangTrain, dataZaiwangTrain])
dataTest = pd.concat([dataLiwangTest, dataZaiwangTest])

print(len(dataTrain))


dataTrain.to_csv("D:\TelecomData\\aa.csv")
dataTest.to_csv("D:\TelecomData\\bb.csv")

data1 = pd.read_csv("D:\TelecomData\\aa.csv")
data2 = pd.read_csv("D:\TelecomData\\bb.csv")
del data1["Unnamed: 0"]
del data2["Unnamed: 0"]
print(data1.head())

for indexs in data1.index:
    # print(data.loc[indexs].values[0:-1])
    # count0 = 0
    flag = 0

    for item in data1.loc[indexs].values[6:7]:
        # if item == 0:
        #     count0 = count0 + 1
        if item == 0:
            flag = 1
    if flag == 1:
        data1.drop([indexs], inplace=True)

print(data2.head())

for indexs in data2.index:
    # print(data.loc[indexs].values[0:-1])
    # count0 = 0
    flag = 0

    for item in data2.loc[indexs].values[6:7]:
        # if item == 0:
        #     count0 = count0 + 1
        if item == 0:
            flag = 1
    if flag == 1:
        data2.drop([indexs], inplace=True)

# data2 = data[data['L0008'].isin([0])]
# print(data2)

# print(data.loc[data['L0008'] == 0].values[0])
data1.to_csv("D:\TelecomData\\train.csv")
data2.to_csv("D:\TelecomData\\test.csv")
