# -*- coding:utf-8 -*-
#导入相关包
import os,sys
import time
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
#from bayes_opt import BayesianOptimization
#from imblearn.over_sampling import SMOTE
#设置路径,读取数据
train_savepath = '/home/hb_liulian/data/train/hbsubs/'
test_savepath = '/home/hb_liulian/data/test/hbsubs/'
model_savepath = '/home/hb_liulian/Model/Model_path/hbsubs/'
result_path = '/home/hb_liulian/result_path/'
df = pd.read_csv(train_savepath + 'train_colrename.csv')

def lookup_feaim0():
    feaim_list = list(base_model.feature_importance())
    feaim0_list = []
    for i in range(len(feaim_list)):
        if list(base_model.feature_importance())[i] ==0:
            feaim0_list.append(i)
        else:
            pass
    return feaim0_list

def str_indexlookup(df):
    list1 = []
    for i in range(df.shape[1]):
        if list(df.dtypes != object)[i]:
            pass
        else:
            list1.append(i)
    use_col = list(df.columns[list(set(list(range(df.shape[1])))-set(list1))])
    return df.loc[:,use_col]

df = str_indexlookup(df)

#label转换
df['LABEL'].loc[df['LABEL']=='True'] = 0
df['LABEL'].loc[df['LABEL']=='False'] = 1

X = df.iloc[:,1:df.shape[1]-1]# 设置训练集特征
y = df.iloc[:,df.shape[1]-1]  # 设置训练集标签
#划分测试机训练集
l_train, l_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=7)
cv_results = 1
#划分测试集训练集
mcw=1/pow(y_train.sum(),0.5)
lgb_train = lgb.Dataset(l_train, y_train, silent=True)
#计算scale_pos_weight
scale_pos_weight = y_train.sum()/y_train.shape[0]
#不平衡样本分层抽样
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)
#计算迭代轮数
def lgb_cv_fit(alg, l_train, y_train, useTrainCV=True,early_stopping_rounds=100):
    if useTrainCV:
        cv_params = alg.get_params()
        global cv_results
        cv_results = lgb.cv(
            cv_params, lgb_train, num_boost_round=alg.get_params()['n_estimators'], nfold=5, 
            stratified=True, shuffle=True, metrics='auc',
            early_stopping_rounds=early_stopping_rounds, verbose_eval=50, show_stdv=True, seed=0)
        n_estimators = len(cv_results['auc-mean'])
        alg.set_params(n_estimators = n_estimators)
    alg.fit(l_train, y_train, eval_metric='logloss')   
    train_predprob = alg.predict_proba(l_train)
    logloss = log_loss(y_train, train_predprob)
bst = lgb.LGBMClassifier(objective='binary'
                            ,num_leaves=32
                            ,learning_rate=0.05
                            ,n_estimators=2000
                            ,max_depth=5
                            ,metric='auc'
                            ,feature_fraction=0.8
                            ,bagging_fraction = 0.8
                            ,bagging_freq = 5
                            ,pos_bagging_fraction = 1
                            ,neg_bagging_fraction = scale_pos_weight
                            ,n_jobs=-1
                            )

print('-------------------开始第一次迭代轮数计算-------------------')
begintime=time.clock()
lgb_cv_fit(bst, l_train, y_train)
endtime=time.clock()
print('best_n_estimators:',len(cv_results['auc-mean']))
print('best_cv_score:', cv_results['auc-mean'][-1])
print ('耗时：',endtime-begintime)
print('-------------------第一次迭代轮数计算结束-------------------')

#挑选无用特征
param = bst.get_params()
base_model=lgb.train(param,lgb_train)
use_fea = list(set(list(range(l_train.shape[1])))-set(lookup_feaim0()))
l_train = pd.DataFrame(l_train).iloc[:,use_fea]

print('-------------------开始第二次迭代轮数计算-------------------')
begintime=time.clock()
lgb_cv_fit(bst, l_train, y_train)
endtime=time.clock()
print('best_n_estimators:',len(cv_results['auc-mean']))
print('best_cv_score:', cv_results['auc-mean'][-1])
print ('耗时：',endtime-begintime)
print('-------------------第二次迭代轮数结束-------------------')

params_test1={
    'n_estimators':[len(cv_results['auc-mean'])]
    ,'learning_rate':np.arange(0.01,0.2,0.01)
    ,'min_child_weight':[mcw]
    #,'min_child_samples': [18,19,20,21,22]
    ,'max_depth': np.arange(3,8,1)
    ,'num_leaves':np.arange(6,260,20)
    ,'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 1.0]
    ,'bagging_freq':[5]
    ,'pos_bagging_fraction':[1]
    ,'neg_bagging_fraction':[scale_pos_weight]
    ,'reg_alpha':np.arange(0,4,0.5)
    ,'reg_lambda':np.arange(0,4,0.5)
}
clf = RandomizedSearchCV(estimator=bst, 
                              param_distributions=params_test1, 
                              scoring='f1', 
                              cv=5, 
                              verbose=1
                              )
#clf = BayesianOptimization(bst,params_test1,random_state=7)
print('-------------------开始搜索最佳超参数-------------------')
begintime=time.clock()
clf.fit(l_train, y_train)
#clf.maximize(init_points=10, n_iter=30, acq='ei', xi=0.0)
endtime=time.clock()
clf.score, clf.best_params_, clf.best_score_#根据设置的测试参数，计算出最优模型参数
print ('耗时：',endtime-begintime)
print('-------------------最佳超参数搜索结束-------------------')

param = clf.best_params_
savemodel=lgb.train(param,lgb_train)
savemodel.save_model(model_savepath+'lgb_subs_model')
bst = lgb.Booster(model_file=model_savepath+'lgb_subs_model')
ypred = bst.predict(l_test, num_iteration=bst.best_iteration)

#对预测结果分层
df1 = pd.DataFrame(ypred)
estimator = KMeans(n_clusters=5)
estimator.fit(df1)
label_pred = estimator.labels_ #获取聚类标签
centroids = estimator.cluster_centers_ #获取聚类中心
df1['Kmeans'] = label_pred
df = pd.DataFrame(df1)
#根据预测概率，从高到低改变聚类标签
for i in range(5):
    df['Kmeans'].ix[df['Kmeans'] ==i]=str(sorted(centroids).index(centroids[i]))+"星"
#计算各个阈值的pre,re,f1
#df['label'] = y_test_smo
df['label'] = y_test.values
df['count'] = 1
df_res=pd.pivot_table(df, index=['Kmeans'],
    values = ['count','label'],
    aggfunc = {'count': np.sum,'label':np.sum})
df_res['precision'] = ''
df_res['recall'] = ''
df_res['实际订购率'] = df_res['label']/df_res['count']
count_list = list(df_res['count'].values)
label_list = list(df_res['label'].values)
for i in range(5):
    df_res['precision'][i] = sum(label_list[i:5])/sum(count_list[i:5])
    df_res['recall'][i] = sum(label_list[i:5])/sum(label_list[0:5])
df_res['f1'] = 2*df_res['precision']*df_res['recall']/(df_res['precision']+df_res['recall'])
df_res['precision'] =df_res['precision'].apply(lambda x: '%.4f%%' % (x*100))
df_res['recall'] =df_res['recall'].apply(lambda x: '%.4f%%' % (x*100))
df_res['f1'] =df_res['f1'].apply(lambda x: '%.4f%%' % (x*100))
df_res['实际订购率'] =df_res['实际订购率'].apply(lambda x: '%.4f%%' % (x*100))
df_res
df_res.to_csv(result_path+'res.csv')

