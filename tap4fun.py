import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import copy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import k_means
from sklearn import linear_model
from sklearn import metrics
from sklearn import cross_validation
def rmse(y_real,y_pre):
    return np.sqrt(metrics.regression.mean_squared_error(y_real,y_pre))

#loading data

test_df=pd.read_csv('tap4fun竞赛数据/tap_fun_test.csv')
train_df=pd.read_csv('tap4fun竞赛数据/tap_fun_train.csv')
len_train=len(train_df)
len_test=len(test_df)
#--------------------------------------------
label=train_df.pop('prediction_pay_price')
x_train=train_df
x_train=x_train.rename(columns={'﻿user_id':'user_id'})
x_test=test_df
#----------------------------------------------
all_data=pd.DataFrame()
all_data=pd.concat([x_train,x_test]).reset_index()
del all_data['index']
del all_data['user_id']

# reduce memory

for col in all_data.columns:
    if all_data[col].dtype=='float64':
        print ('transform to float32 '+col)
        all_data[col]= all_data[col].astype('float32')
    elif all_data[col].dtype=='int64':
        print ('transform to int32 '+ col)
        all_data[col]= all_data[col].astype('int32')


#featuren engeering


# In[52]:

'''
all_data['year']=all_data.register_time.apply(lambda x:x.split(' ')[0]).apply(lambda x:x.split('-')[0]).astype('int')
all_data['month']=all_data.register_time.apply(lambda x:x.split(' ')[0]).apply(lambda x:x.split('-')[1]).astype('int')
all_data['day']=all_data.register_time.apply(lambda x:x.split(' ')[0]).apply(lambda x:x.split('-')[2]).astype('int')
all_data['hour']=all_data.register_time.apply(lambda x:x.split(' ')[1]).apply(lambda x:x.split(':')[0]).astype('int')

'''
del all_data['register_time']


# In[57]:

def pvp_active_ratio(df):
    if df['pvp_battle_count'] != 0:
        return df['pvp_lanch_count']/df['pvp_battle_count']
    else:
        return 0


# In[58]:

all_data['pvp_active_ratio']=all_data.apply(pvp_active_ratio,axis=1).astype('float32')


# In[59]:

def pvp_win_ratio(df):
    if df['pvp_battle_count'] != 0:
        return df['pvp_win_count']/df['pvp_battle_count']
    else:
        return 0


# In[60]:

all_data['pvp_win_ratio']=all_data.apply(pvp_win_ratio,axis=1).astype('float32')


# In[61]:

def pve_active_ratio(df):
    if df['pve_battle_count'] != 0:
        return df['pve_lanch_count']/df['pve_battle_count']
    else:
        return 0


# In[62]:

all_data['pve_active_ratio']=all_data.apply(pve_active_ratio,axis=1).astype('float32')


# In[63]:

def pve_win_ratio(df):
    if df['pve_battle_count'] != 0:
        return df['pve_win_count']/df['pve_battle_count']
    else:
        return 0


# In[64]:

all_data['pve_win_ratio']=all_data.apply(pve_win_ratio,axis=1).astype('float32')


# In[65]:

def avg_pay_price(df):
    if df['pay_count']!=0:
        return all_data['pay_price']/all_data['pay_count']
    else :
        return 0
    
all_data.apply(avg_pay_price,axis=1).astype('float32')

# In[66]:

all_data.columns


# In[67]:

def wood_active_ratio(df):
    if df['wood_add_value'] != 0:
        return df['wood_reduce_value']/df['wood_add_value']
    else:
        return 0
all_data['wood_active_ratio']=all_data.apply(wood_active_ratio,axis=1).astype('float32')
all_data['wood_available']=all_data.apply(lambda x:x['wood_add_value']-x['wood_reduce_value'],axis=1).astype('float32')


# In[68]:

def training_acceleration_active_ratio(df):
    if df['training_acceleration_add_value'] != 0:
        return df['training_acceleration_reduce_value']/df['training_acceleration_add_value']
    else:
        return 0
all_data['training_acceleration_active_ratio']=all_data.apply(training_acceleration_active_ratio,axis=1).astype('float32')
all_data['training_acceleration_available']=all_data.apply(lambda x:x['training_acceleration_add_value']-x['training_acceleration_reduce_value'],axis=1).astype('float32')


# In[69]:

#石头

def stone_active_ratio(df):
    if df['stone_add_value'] != 0:
        return df['stone_reduce_value']/df['stone_add_value']
    else:
        return 0
all_data['stone_active_ratio']=all_data.apply(stone_active_ratio,axis=1).astype('float32')
all_data['stone_available']=all_data.apply(lambda x:x['stone_add_value']-x['stone_reduce_value'],axis=1).astype('float32')


# In[70]:

#象牙
def ivory_active_ratio(df):
    if df['ivory_add_value'] != 0:
        return df['ivory_reduce_value']/df['ivory_add_value']
    else:
        return 0
all_data['ivory_active_ratio']=all_data.apply(ivory_active_ratio,axis=1).astype('float32')
all_data['ivory_available']=all_data.apply(lambda x:x['ivory_add_value']-x['ivory_reduce_value'],axis=1).astype('float32')


# In[71]:

#肉
def meat_active_ratio(df):
    if df['meat_add_value'] != 0:
        return df['meat_reduce_value']/df['meat_add_value']
    else:
        return 0
all_data['meat_active_ratio']=all_data.apply(meat_active_ratio,axis=1).astype('float32')
all_data['meat_available']=all_data.apply(lambda x:x['meat_add_value']-x['meat_reduce_value'],axis=1).astype('float32')


# In[72]:

#魔法
def magic_active_ratio(df):
    if df['magic_add_value'] != 0:
        return df['magic_reduce_value']/df['meat_add_value']
    else:
        return 0
all_data['magic_active_ratio']=all_data.apply(magic_active_ratio,axis=1).astype('float32')
all_data['magic_available']=all_data.apply(lambda x:x['magic_add_value']-x['magic_reduce_value'],axis=1).astype('float32')


# 勇士
def infantry_active_ratio(df):
    if df['infantry_add_value'] != 0:
        return df['infantry_reduce_value']/df['infantry_add_value']
    else:
        return 0
all_data['infantry_active_ratio']=all_data.apply(infantry_active_ratio,axis=1).astype('float32')
all_data['infantry_available']=all_data.apply(lambda x:x['infantry_add_value']-x['infantry_reduce_value'],axis=1).astype('float32')

#驯兽师
def cavalry_active_ratio(df):
    if df['cavalry_add_value'] != 0:
        return df['cavalry_reduce_value']/df['cavalry_add_value']
    else:
        return 0
all_data['cavalry_active_ratio']=all_data.apply(cavalry_active_ratio,axis=1).astype('float32')
all_data['cavalry_available']=all_data.apply(lambda x:x['cavalry_add_value']-x['cavalry_reduce_value'],axis=1).astype('float32')

#萨满
def shaman_active_ratio(df):
    if df['shaman_add_value'] != 0:
        return df['shaman_reduce_value']/df['shaman_add_value']
    else:
        return 0
all_data['shaman_active_ratio']=all_data.apply(shaman_active_ratio,axis=1).astype('float32')
all_data['shaman_available']=all_data.apply(lambda x:x['shaman_add_value']-x['shaman_reduce_value'],axis=1).astype('float32')



# analysis corr

corr_columns=[]
for col in all_data.columns:   
    corr = all_data[col].corr(label)
    if corr >0.2:
        corr_columns.append(col)
        print (col,corr)



#分成三个数据集的方法

#x_train, x_val, y_train, y_val = train_test_split(all_data.ix[:len_train-1], label, test_size=0.33, random_state=42)
#pd.read_csv('distance_list.csv').distance_list.values.tolist()
print ('creat validation set .....')
val_index=pd.read_csv('distance_list.csv').distance_list.values.tolist()
y_val=label[val_index]
dval=xgb.DMatrix(all_data.ix[val_index],label=y_val)

print ('creat  train set.....')

train_index=set(train_df.index.values)-set(val_index)
len_train_part= len(train_index)

part_len=int(len_train_part/3)

y_train=label[train_index]

dtrain1=xgb.DMatrix(all_data.ix[:part_len-1],label=y_train[:part_len])
dtrain2=xgb.DMatrix(all_data.ix[part_len:2*part_len-1],label=y_train[part_len:2*part_len])
dtrain3=xgb.DMatrix(all_data.ix[2*part_len:],label=y_train[2*part_len:])

print ('creat  test set.....')

dtest=xgb.DMatrix(all_data.ix[len_train:])


#model1

param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'reg:linear','gamma':0.1}
num_round = 27
evalist=[(dtrain1,'train'),(dval,'val')]
bst1 = xgb.train(param, dtrain1, num_round,evalist)
result_df1=pd.DataFrame()
result_df1['user_id']=test_df.user_id.values
result_df1['prediction_pay_price']=bst1.predict(dtest)

#model 2

param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'reg:linear','min_child_weight':0.5,'lambda':0.5}
num_round = 30
evalist=[(dtrain2,'train'),(dval,'val')]
bst2 = xgb.train(param, dtrain2, num_round,evalist)
result_df2=pd.DataFrame()
result_df2['user_id']=test_df.user_id.values
result_df2['prediction_pay_price']=bst2.predict(dtest)

#model3
param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'reg:linear','colsample_bytree':0.8,'alpha':0.5}
num_round = 30
evalist=[(dtrain3,'train'),(dval,'val')]
bst3 = xgb.train(param, dtrain3, num_round,evalist)
result_df3=pd.DataFrame()
result_df3['user_id']=test_df.user_id.values
result_df3['prediction_pay_price']=bst.predict(dtest)