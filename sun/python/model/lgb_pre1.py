# -*- coding:utf-8 -*-
#导入相关包
print('----------------------------------------------------------------初始化------------------------------------------------------------------------')
import os,sys
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.cluster import KMeans
test_savepath = '/home/hb_liulian/data/test/hbsubs/'
model_savepath = '/home/hb_liulian/model/hbsubs/'
result_path = '/home/hb_liulian/result/'
col_name = [
'USER_ID',
'MSISDN',
'CUST_ID',
'ACCT_ID',
'CITY_CODE',
'AREA_ID',
'USER_TYPE',
'INNET_DATE',
'INNET_MONTH',
'SERVICE_TYPE',
'INNET_CHNL_ID',
'INNET_CHNL_TYPE',
'PAY_MODE',
'INNET_METHOD',
'BRAND_ID',
'USER_STAR_LVL',
'DATA_SIMCARD_M2M_USER_FLAG',
'DATA_SIMCARD_M2M_USER_FLAG_YX',
'DATA_SIMCARD_M2M_USER_FLAG_TD',
'USER_STATUS_TYPE',
'STATUS_EFF_DATE',
'GENDER',
'AGE',
'CREDENTIALS_TYPE',
'ONECARD_MSISDN_CNT',
'EXPENSES_PLAN_CODE',
'MAIN_PLAN_CODE',
'MAIN_PLAN_LVL',
'LAST_MAIN_PLAN_CODE',
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
'IS_THIS_INNET',
'IS_THIS_BREAK',
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
'IMEI',
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
'ENT_UNIPAY_INCOME',
'THIS_ACCT_FEE_TAX',
'TAX_LOCAL_VOICE_FEE',
'TAX_ROAM_VOICE_FEE',
'TAX_LONG_VOICE_FEE',
'TAX_RENT_FEE',
'TAX_VAS_FEE',
'TAX_DATA_FEE',
'TAX_CMNET_FEE',
'TAX_FAV_FEE',
'TAX_OTHER_FEE',
'THIS_ACCT_FEE_NOTAX',
'NOTAX_LOCAL_VOICE_FEE',
'NOTAX_ROAM_VOICE_FEE',
'NOTAX_LONG_VOICE_FEE',
'NOTAX_RENT_FEE',
'NOTAX_VAS_FEE',
'NOTAX_DATA_FEE',
'NOTAX_CMNET_FEE',
'NOTAX_FAV_FEE',
'NOTAX_OTHER_FEE',
'T_1_ACCT_FEE_TAX',
'T_2_ACCT_FEE_TAX',
'T_3_ACCT_FEE_TAX',
'T_4_ACCT_FEE_TAX',
'T_5_ACCT_FEE_TAX',
'T_6_ACCT_FEE_TAX',
'PAY_FEE',
'PAY_TIMES',
'PAY_TIMES_L6M',
'PAY_FEE_L6M',
'PAY_DATE_LAST',
'THIS_OWE_FEE',
'OWE_FEE',
'IS_THIS_ACCT',
'IS_THIE_OWE',
'AVG_ARPU_L3M',
'IS_THIS_ACCT_RETAIN',
'VOICE_DURA',
'OUT_VOICE_DURA',
'NOROAM_OUT_VOICE_DURA',
'ROAM_OUT_VOICE_DURA',
'IN_VOICE_DURA',
'NOROAM_IN_VOICE_DURA',
'ROAM_IN_VOICE_DURA',
'JF_TIMES',
'JF_TIMES_2G',
'JF_TIMES_3G',
'JF_TIMES_4G',
'OUT_VOICE_JF_TIMES',
'OUT_LOCAL_JF_TIMES',
'OUT_GN_LONG_JF_TIMES',
'NOROAM_JF_TIMES',
'NOROAM_LOCAL_JF_TIMES',
'ROROAM_LONG_JF_TIMES',
'NOROAM_SN_JF_TIMES',
'NOROAM_SJ_JF_TIMES',
'NOROAM_GAT_JF_TIMES',
'NOROAM_GJ_JF_TIMES',
'ROAM_JF_TIMES',
'ROAM_SN_JF_TIMES',
'ROAM_SN_CALL_GN_JF_TIMES',
'ROAM_SN_CALL_GAT_JF_TIMES',
'ROAM_SN_CALL_GJ_JF_TIMES',
'ROAM_SJ_JF_TIMES',
'ROAM_SJ_CALL_GN_JF_TIMES',
'ROAM_SJ_CALL_GAT_JF_TIMES',
'ROAM_SJ_CALL_GJ_JF_TIMES',
'ROAM_GAT_JF_TIMES',
'ROAM_GJ_JF_TIMES',
'IN_JF_TIMES',
'VOICE_CNT',
'VOICE_CNT_2G',
'VOICE_CNT_3G',
'VOICE_CNT_4G',
'LOCAL_VOICE_CNT',
'OUT_VOICE_CNT',
'NOROAM_OUT_VOICE_CNT',
'ROAM_OUT_VOICE_CNT',
'IN_VOICE_CNT',
'NOROAM_IN_VOICE_CNT',
'ROAM_IN_VOICE_CNT',
'SMS_CNT',
'IS_VOICE_FLAG',
'IS_TOLL_VOICE_FLAG',
'IS_ROAM_VOICE_FLAG',
'VIDEO_JF_TIMES_3G',
'VIDEO_JF_TIMES_4G',
'P2P_SMS_CNT',
'MNET_SMS_CNT',
'MMS_CNT',
'P2P_MMS_CNT',
'MNET_MMS_CNT',
'NEWSPAPER_MMS_CNT',
'GPRS_2G_FLUX_UP',
'GPRS_2G_FLUX_DOWN',
'GPRS_2G_FLUX_DURA',
'GPRS_2G_FLUX_CNT',
'GPRS_3G_FLUX_UP',
'GPRS_3G_FLUX_DOWN',
'GPRS_3G_FLUX_DURA',
'GPRS_3G_FLUX_CNT',
'GPRS_4G_FLUX_UP',
'GPRS_4G_FLUX_DOWN',
'GPRS_4G_FLUX_DURA',
'GPRS_4G_FLUX_CNT',
'GPRS_VOLTE_FLUX_UP',
'GPRS_VOLTE_FLUX_DOWN',
'GPRS_VOLTE_FLUX_DURA',
'GPRS_VOLTE_FLUX_CNT',
'GPRS_TOTAL_FLUX',
'WLAN_FLUX_UP',
'CMWAP_FLUX',
'CMNET_FLUX',
'WLAN_FLUX_DOWN',
'WLAN_FLUX_DURA',
'WLAN_FLUX_CNT',
'OVER_PLAN_FEE',
'FLUX_FEE_TAX',
'SMS_FEE_TAX',
'VOICE_FEE_TAX',
'OVER_FEE_TAX',
'FLUX_FEE_NOTAX',
'SMS_FEE_NOTAX',
'VOICE_FEE_NOTAX',
'VOICE_DURA_L1M',
'VOICE_DURA_L2M',
'VOICE_DURA_L3M',
'VOICE_DURA_L4M',
'VOICE_DURA_L5M',
'VOICE_DURA_L6M',
'FLUX_L1M',
'FLUX_L2M',
'FLUX_L3M',
'FLUX_L4M',
'FLUX_L5M',
'FLUX_L6M',
'SMS_CNT_L1M',
'SMS_CNT_L2M',
'SMS_CNT_L3M',
'SMS_CNT_L4M',
'SMS_CNT_L5M',
'SMS_CNT_L6M',
'IS_THIS_ACTIVE',
'IS_THIS_W3',
'IS_THIS_SILENT',
'IS_THIS_NOMINAL',
'THIS_CALL_DAYS',
'THIS_FLUX_DAYS',
'IS_MKT_CASE_USER',
'LAST_MKT_CASE_KIND',
'LAST_MKT_CASE_EFF_DATE',
'LAST_MKT_CASE_EXP_DATE',
'USER_ENABLE_SCORE',
'THIS_EXCH_SCORE',
'GPRS_PACK_FEE',
'L1M_GPRS_PACK_FEE',
'L2M_GPRS_PACK_FEE',
'L3M_GPRS_PACK_FEE',
'GPRS_COMM_FEE',
'L1M_GPRS_COMM_FEE',
'L2M_GPRS_COMM_FEE',
'L3M_GPRS_COMM_FEE',
'IS_DOUBLE_REDUCE',
'is_diff_net_user',
'two_imei_user_cnt',
'CAncel_acct_fee',
'reward_fee',
'ptp_sms_oppo_cnt',
'voice_out_oppo_cnt',
'voice_in_oppo_cnt',
'voice_basesta_cnt',
'flux_basesta_cnt',
'STATIS_DATE',
'PROV_ID',
]

def str_indexlookup(df):
    list1 = []
    for i in range(df.shape[1]):
        if list(df.dtypes != object)[i]:
            pass
        else:
            list1.append(i)
    use_col = list(df.columns[list(set(list(range(df.shape[1])))-set(list1))])
    return df.loc[:,use_col]

print('---------------------------------------------------------开始读取数据-----------------------------------------------------------------------')	
df = pd.read_csv(test_savepath+sys.argv[1],low_memory=False,header = 0 ,names = col_name)
print('---------------------------------------------------------数据读取完毕-----------------------------------------------------------------------')
print('---------------------------------------------------------剔除str字段------------------------------------------------------------------------')
df = str_indexlookup(df)
print('-----------------------------------------------------------剔除完毕-------------------------------------------------------------------------')

l_test = df.iloc[:,1:df.shape[1]-1]
print('------------------------------------------------------------加载模型文件--------------------------------------------------------------------')
bst = lgb.Booster(model_file=model_savepath+sys.argv[2])
print('--------------------------------------------------------------开始预测----------------------------------------------------------------------')
ypred = bst.predict(l_test, num_iteration=bst.best_iteration)
#对预测结果分层
df1 = pd.DataFrame(ypred)
estimator = KMeans(n_clusters=5)
print('--------------------------------------------------------------结果聚类----------------------------------------------------------------------')
estimator.fit(df1)
label_pred = estimator.labels_ #获取聚类标签
centroids = estimator.cluster_centers_ #获取聚类中心
df1['user_id'] = df['user_id']
df1['Kmeans'] = label_pred
df1 = df1.loc[df1['Kmeans']==4]
print('--------------------------------------------------------------输出结果----------------------------------------------------------------------')
df1.to_csv(result_path+sys.argv[3])
print('----------------------------------------------------------------END-------------------------------------------------------------------------')

