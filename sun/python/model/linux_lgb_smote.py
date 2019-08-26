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
from imblearn.over_sampling import SMOTE
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

def lgb_cv_fit(alg,useTrainCV=True):
    if useTrainCV:
        cv_params = alg.get_params()
        global cv_results
        cv_results = lgb.cv(
            cv_params, lgb_train, num_boost_round=alg.get_params()['num_iterations'], nfold=5, 
            stratified=True, shuffle=True, metrics='binary_logloss',
            early_stopping_rounds=50, verbose_eval=50, show_stdv=True, seed=0)
        num_iterations = len(cv_results['binary_logloss-mean'])

#设置类别型特征
catfea_list = [
'CITY_CODE',
'AREA_ID',
'USER_TYPE',
'SERVICE_TYPE',
'PAY_MODE',
'INNET_METHOD',
'BRAND_ID',
'USER_STAR_LVL',
'USER_STATUS_TYPE',
'GENDER',
'AGE',
'CREDENTIALS_TYPE',
'ONECARD_MSISDN_CNT',
'IS_PRIME_GOTONE',
'IS_BROAD_BIND',
'IS_UNLIM_PLAN',
'IS_CM_UNLIM_EFF',
'MAIN_ASSI_USER',
'ASSI_MAIN_USER_ID',
'IS_MAIN_ASSI',
'ORD_BUSI_TYPE',
'ASSI_TYPE',
'IS_THIS_DEV',
'ID_BREAK_TYPE',
'IS_SAKA',
'IS_DM_SEEP',
'IS_BROAD_SEEP',
'IS_THIS_HALT',
'IS_GROUP',
'IS_GROUP_IMP',
'MEM_TYPE',
'IS_SCHOOL_USER',
'IS_SCHOOL_AREA_USER',
'IS_FAM_VNET',
'IS_DEV_SHAM',
'IS_FLEA',
'IS_FLUX_SHAM',
'IS_LOW_STATUS_EXCEPT',
'IS_LOW_ACTIVE',
'IS_THIS_CHANGE_PLAN',
'TERM_OS',
'IS_DOUBLE_CARD',
'DOUBLE_CARD_TYPE',
'IS_RAISE_CARD',
'RAISE_CARD_TYPE',
'ONE_IMEI_SHARE_TYPE',
'IS_DOUBLE_IMEI',
'IMSI1_USER_TYPE',
'IMSI2_USER_TYPE',
'BASE_PLAN_PRICE',
'IS_4G_USER',
'IS_4G_OPEN',
'TERM_4G_CUST_FLAG',
'MOBILE_4G_CUST_FLAG',
'MIFI_4G_CUST_FLAG',
'CPE_4G_CUST_FLAG',
'CARD_4G_CUST_FLAG',
'USE_NET_4G_CUST_FLAG',
'IS_FAV_FEE',
'IS_THIS_ACCT',
'IS_THIE_OWE',
'IS_THIS_ACCT_RETAIN',
'IS_VOICE_FLAG',
'IS_TOLL_VOICE_FLAG',
'IS_ROAM_VOICE_FLAG',
'IS_THIS_ACTIVE',
'IS_THIS_W3',
'IS_THIS_SILENT',
'IS_THIS_NOMINAL',
'IS_MKT_CASE_USER',
'LAST_MKT_CASE_KIND',
'IS_DOUBLE_REDUCE',
'is_diff_net_user',
'two_imei_user_cnt'
]

df = str_indexlookup(df)
df = df.dropna(axis = 1,thresh = df.shape[0]*0.6)
df = df.fillna(method='ffill')
#label转换
df['LABEL'].loc[df['LABEL']==True] = 'a'
df['LABEL'].loc[df['LABEL']==False] = 'b'
df['LABEL'].loc[df['LABEL']=='a'] = 0
df['LABEL'].loc[df['LABEL']=='b'] = 1

X = df.iloc[:,1:df.shape[1]-1]# 设置训练集特征
y = df.iloc[:,df.shape[1]-1]  # 设置训练集标签
#划分测试机训练集
l_train, l_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=7)

#对训练集数据用SMOTE算法合成pos_sample
smo = SMOTE(ratio={1: round(y_train.shape[0]/4)},random_state=42)
X_smo, y_smo = smo.fit_sample(l_train, y_train)
df_xsmo = pd.DataFrame(X_smo)
df_ysmo = pd.DataFrame(y_smo)
df_xsmo_renamedict = dict(zip(df_xsmo.columns.values.tolist(),X.columns.values.tolist()))
df_xsmo.rename(columns = df_xsmo_renamedict,inplace=True)
df_ysmo.rename(columns = {0:'label'},inplace=True)

#定义一些变量
cv_results = 1
mcw=1/pow(y_smo.sum(),0.5)
scale_pos_weight = y_smo.sum()/y_smo.shape[0]
#生成第一次CV需要的lgb_train
lgb_train = lgb.Dataset(df_xsmo, label = df_ysmo, silent=True,
                        feature_name = df_xsmo.columns.values.tolist(),
                        categorical_feature = cat_list)

#不平衡样本分层抽样
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)
#设置lgb模型
bst = lgb.LGBMClassifier(objective='binary'
                            ,num_leaves=32
                            ,learning_rate=0.05
                            ,num_iterations=2000
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
begintime=time.perf_counter()
lgb_cv_fit(bst, X_smo, y_smo)
endtime=time.perf_counter()
print('best_num_iterations:',len(cv_results['auc-mean']))
print('best_cv_score:', cv_results['auc-mean'][-1])
print ('耗时：',endtime-begintime)
print('-------------------第一次迭代轮数计算结束-------------------')

#挑选无用特征
param = bst.get_params()
lgb_train = lgb.Dataset(df_xsmo, label = df_ysmo, silent=True,
                        feature_name = df_xsmo.columns.values.tolist(),
                        categorical_feature = cat_list)
base_model=lgb.train(param,lgb_train)
use_fea = list(set(list(range(l_train.shape[1])))-set(lookup_feaim0()))
X_refea = X.iloc[:,use_fea]# 设置剔除无用特征训练集特征
y_refea = df.iloc[:,df.shape[1]-1] # 设置训练集标签
#对训练集数据用SMOTE算法合成pos_sample
smo = SMOTE(ratio={1: round(y_train.shape[0]/4)},random_state=42)
X_smo_refea, y_smo_refea = smo.fit_sample(X_refea, y_refea)
df_xsmo_refea = pd.DataFrame(X_smo_refea)
df_ysmo_refea = pd.DataFrame(y_smo_refea)
df_xsmo_refea_renamedict = dict(zip(df_xsmo_refea.columns.values.tolist(),X_refea.columns.values.tolist()))
df_xsmo_refea.rename(columns = df_xsmo_refea_renamedict,inplace=True)
df_ysmo_refea.rename(columns = {0:'label'},inplace=True)
cat_refea_list = list(set(cat_list) - set(X.iloc[:,lookup_feaim0()].columns.values.tolist()))
#生成剔除无用特征后的lgb_train
lgb_train = lgb.Dataset(df_xsmo_refea, label = df_ysmo_refea, 
                        silent=True,
                        feature_name = df_xsmo_refea.columns.values.tolist(),
                        categorical_feature = cat_refea_list)

print('-------------------开始第二次迭代轮数计算-------------------')
begintime=time.perf_counter()
lgb_cv_fit(bst)
endtime=time.perf_counter()
print('best_n_estimators:',len(cv_results['auc-mean']))
print('best_cv_score:', cv_results['auc-mean'][-1])
print ('耗时：',endtime-begintime)
print('-------------------第二次迭代轮数计算结束-------------------')
#设置超参数空间
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
begintime=time.perf_counter()
clf.fit(df_xsmo_refea, df_ysmo_refea)
#clf.maximize(init_points=10, n_iter=30, acq='ei', xi=0.0)
endtime=time.perf_counter()
clf.score, clf.best_params_, clf.best_score_#根据设置的测试参数，计算出最优模型参数
print ('耗时：',endtime-begintime)
print('-------------------最佳超参数搜索结束-------------------')
#设置最终模型需要的lgb_train
lgb_train_refea = lgb.Dataset(df_xsmo_refea, label = df_ysmo_refea, 
                        silent=True,
                        feature_name = df_xsmo_refea.columns.values.tolist(),
                        categorical_feature = cat_refea_list)
#设置剔除无用特征的l_test
l_test = l_test.iloc[:,use_fea]
#对验证集进行预测
param = clf.best_params_
savemodel=lgb.train(param,lgb_train)
savemodel.save_model(model_savepath+'lgb_subs_model_smo')
bst = lgb.Booster(model_file=model_savepath+'lgb_subs_model_smo')
ypred = bst.predict(l_test, num_iteration=bst.best_iteration)
df_ypred = pd.DataFrame(ypred)

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
print(df_res)

#输出结果
df_res.to_csv(result_path+'res_smote.csv')
df_ypred.to_csv(result_path+'res_pre.csv')

