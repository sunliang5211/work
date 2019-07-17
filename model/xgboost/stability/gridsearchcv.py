# -*- coding:utf-8 -*-
#导入相关包
import os,sys
import time
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
#设置路径,读取数据
train_savepath = 'D:/data/cox/'
model_savepath = 'D:/work/xgboost/stability/'
train = pd.read_csv(train_savepath + 'inner5&6.csv')

#定义全局变量
cvresult = 1
X_train = train.iloc[:,1:train.shape[1]-1]# 设置训练集特征
y_train = train.iloc[:,train.shape[1]-1]  # 设置训练集标签
#计算min_child_weight参数的值，较少的y开根号分至一
mcw=1/pow(pd.value_counts(y_train)[1],0.5)
#不平衡样本分层抽样
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)
#确定弱学习器数量的函数
def modelfit(alg, X_train, y_train, useTrainCV=True, cv_folds=None, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(X_train, label = y_train)
        global cvresult 
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], folds =cv_folds,
             metrics='logloss', early_stopping_rounds=early_stopping_rounds)     
        n_estimators = cvresult.shape[0]
        alg.set_params(n_estimators = n_estimators)
    alg.fit(X_train, y_train, eval_metric='logloss')   
    train_predprob = alg.predict_proba(X_train)
    logloss = log_loss(y_train, train_predprob)
#默认参数设置个模型,然后跑出CV确定树的数量
bst = xgb.XGBClassifier(
        learning_rate =0.1,
        n_estimators=1000,  #数值大没关系，cv会自动返回合适的n_estimators
        max_depth=4,
        min_child_weight=mcw,
        gamma=0,
        subsample=0.8,
        colsample_bytree=1,
        colsample_bylevel=0.8,
        objective='binary:logistic',
        n_jobs=8,
        #tree_method='gpu_hist',
        #gpu_id=1,
        seed=3)

begintime=time.clock()
modelfit(bst, X_train, y_train, cv_folds = kfold)
endtime=time.clock()
print (endtime-begintime)

cvresult.shape[0]

param_test = {
'n_estimators':[cvresult.shape[0]]
,'min_child_weight':[mcw]#np.arange(202,302,10)
,'max_depth':[4,5,6]
,'learning_rate':np.arange(0.01,0.1,0.02)
#,'scale_pos_weight':[0.08,0.25,0.5,0.75,1]
,'subsample':[0.85]#np.arange(0.6,0.9,0.05)
#,'colsample_bytree':[1]
,'silent':[1]
,'reg_alpha':[1.5]#np.arange(0,4,0.5)
,'reg_lambda':[0.5]#np.arange(0,4,0.5)
,'gamma':[0]#np.arange(1.6,2.6,0.1)
,'seed':[3]
}#设置需要进行测试的参数，参数设置慎重，数据量过大，运算会非常慢
clf = GridSearchCV(estimator = bst, param_grid = param_test, scoring='neg_log_loss', n_jobs=8,cv=5)
#clf = GridSearchCV(estimator = bst, param_grid = param_test, scoring='f1', n_jobs=8,cv=5)
begintime=time.clock()
clf.fit(X_train, y_train)
endtime=time.clock()
clf.score, clf.best_params_, clf.best_score_#根据设置的测试参数，计算出最优模型参数
print (endtime-begintime)

clf.score, clf.best_params_, clf.best_score_


param = clf.best_params_
savemodel=xgb.XGBClassifier(**param)
savemodel.fit(X_train,y_train)
savemodel.save_model(model_savepath+'xgb_model_hblost')

# plot feature importance using built-in function
from xgboost import plot_importance
from matplotlib import pyplot
pyplot.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
pyplot.rcParams['axes.unicode_minus']=False #用来正常显示负号
plot_importance(bst)
pyplot.show()

