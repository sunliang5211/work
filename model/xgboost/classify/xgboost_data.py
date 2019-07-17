import xgboost as xgb
import operator
import matplotlib.pyplot as plt

dtrain = xgb.DMatrix('train.svm.txt')
dtest = xgb.DMatrix('test.svm.txt')

params={
'booster':'gbtree',
'objective': 'multi:softmax', #多分类问题
'num_class':10, # 类数，与 multisoftmax 并用
'gamma':0.05,  # 在树的叶子节点下一个分区的最小损失，越大算法模型越保守 。[0:]
'max_depth':12, # 构建树的深度 [1:]
#'lambda':2,  # L2 正则项权重
'subsample':1, # 采样训练数据，设置为0.5，随机选择一般的数据实例 (0:1]
'colsample_bytree':0.5, # 构建树时的采样比率 (0:1]
#'min_child_weight':3, # 节点的最少特征数
'silent':1 ,
'eta': 0.005, # 如同学习率
'seed':710,
'nthread':4,# cpu 线程数,根据自己U的个数适当调整
}
plst = list(params.items())
num_rounds = 5
evallist = [(dtest, 'test'), (dtrain, 'train')]

model = xgb.train(plst, dtrain, num_rounds, evallist,early_stopping_rounds=100)
model.save_model("0001.model")

importance = model.get_fscore()
print (len(importance))
importance = sorted(importance.items(), key=operator.itemgetter(1))
print (importance)

xgb.plot_importance(model,max_num_features=50)
plt.show()