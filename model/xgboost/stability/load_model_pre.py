
# coding: utf-8

# In[1]:


#加载相关包
from xgboost import XGBClassifier
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.cluster import KMeans


# In[4]:


#设置测试集数据、模型文件、输入结果路径
test_data_path = 'D:\work\xgboost\stability\'
model_path = 'D:\work\xgboost\stability\'
result_path = 'D:\work\xgboost\stability\'
test_file_path = test_data_path+"test.xlsx"
xgbmodel_file_path = model_path+"xgb_model_hblost"
result_file_path = result_path+"result.xlsx"


# In[5]:


#读取预测数据
test = pd.read_excel(test_file_path)


# In[6]:


#设置xgb模型的训练集
xgb_test = test.iloc[:,1:test.shape[1]-1]#iloc函数选取测试集第2列(python里通常情况下0代表第一列)到最后一列为特征
dX_test = xgb.DMatrix(xgb_test)#使用xgboost包自带的函数转换为xgboost模型需要的数据格式
#设置LR模型的训练集
#lr_test = test.iloc[:,1:test.shape[1]].fillna(0) 


# In[7]:


#加载保存的m模型
bst_new = xgb.Booster(model_file=xgbmodel_file_path)
#lr_new = joblib.load(lrmodel_file_path)


# In[8]:


#使用xgb模型对测试集进行预测
xgb_preds = pd.DataFrame(bst_new.predict(dX_test))[0]
#lr_preds = pd.DataFrame(lr_new.predict_proba(lr_test))[1]
preds =xgb_preds#*0.6+lr_preds*0.4
test["pre"]=preds


# In[10]:


#Kmeans聚类客户层次
df1 = pd.DataFrame(test["pre"])
estimator = KMeans(n_clusters=5)
estimator.fit(df1)
label_pred = estimator.labels_ #获取聚类标签
centroids = estimator.cluster_centers_ #获取聚类中心
test["Kmeans"] = label_pred
df = pd.DataFrame(test[["user_id","pre","Kmeans"]])
#根据预测概率，从高到低改变聚类标签
for i in range(5):
    df['Kmeans'].ix[df['Kmeans'] ==i]=str(sorted(centroids).index(centroids[i]))+"星"


# In[11]:


#输出预测结果
df.to_excel(result_file_path)


# In[10]:


#查看各个输入特征的重要性
from xgboost import plot_importance
import matplotlib
from matplotlib import pyplot
matplotlib.rcParams['font.sans-serif'] = ['SimHei'] 
matplotlib.rcParams['font.family']='sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False
plot_importance(bst_new)
pyplot.show()

