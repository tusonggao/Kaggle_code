import pickle
import time
import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xgboost as xgb
import lightgbm as lgb

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer



train_df = pd.read_csv('C:/D_Disk/data_competition/home_credit/all/application_train.csv', index_col=0, header=0)
test_df = pd.read_csv('C:/D_Disk/data_competition/home_credit/all/application_test.csv', index_col=0, header=0)
test_df['TARGET'] = -100
merged_df = train_df.append(test_df)

merged_df = merged_df[merged_df['CODE_GENDER'] != 'XNA']
    
# NaN values for DAYS_EMPLOYED: 365.243 -> nan
merged_df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
# Some simple new features (percentages)
merged_df['DAYS_EMPLOYED_PERC'] = merged_df['DAYS_EMPLOYED'] / merged_df['DAYS_BIRTH']
merged_df['INCOME_CREDIT_PERC'] = merged_df['AMT_INCOME_TOTAL'] / merged_df['AMT_CREDIT']
merged_df['INCOME_PER_PERSON'] = merged_df['AMT_INCOME_TOTAL'] / merged_df['CNT_FAM_MEMBERS']
merged_df['ANNUITY_INCOME_PERC'] = merged_df['AMT_ANNUITY'] / merged_df['AMT_INCOME_TOTAL']
merged_df['PAYMENT_RATE'] = merged_df['AMT_ANNUITY'] / merged_df['AMT_CREDIT']

pos_balance_df = pd.read_csv('C:/D_Disk/data_competition/home_credit/all/POS_CASH_balance_outcome_df.csv', index_col=0, header=0)
pos_balance_df.rename(columns=lambda x: 'pos_balance_' + x, inplace=True)    
merged_df = merged_df.join(pos_balance_df, how='left')
del pos_balance_df
gc.collect()
print('after merged pos_balance_df merged_df.shape is ', merged_df.shape)


bureau_outcome_df = pd.read_csv('C:/D_Disk/data_competition/home_credit/bureau_outcome_df_v3.csv', index_col=0, header=0)
bureau_outcome_df.rename(columns=lambda x: 'bureau_outcome_' + x, inplace=True)    
merged_df = merged_df.join(bureau_outcome_df, how='left')
del bureau_outcome_df
gc.collect()
print('after merged bureau_outcome_df merged_df.shape is ', merged_df.shape)

credit_card_balance_outcome_df = pd.read_csv('C:/D_Disk/data_competition/home_credit/credit_card_balance_outcome_df_v1.csv',
                                             index_col=0, header=0)
merged_df = merged_df.join(credit_card_balance_outcome_df, how='left')
del credit_card_balance_outcome_df
gc.collect()
print('after merged credit_card_balance_outcome_df merged_df.shape is ', merged_df.shape)


installments_payments_outcome_df = pd.read_csv('C:/D_Disk/data_competition/home_credit/installments_payments_outcome_df_v1.csv', 
                                               index_col=0, header=0)
merged_df = merged_df.join(installments_payments_outcome_df, how='left')
del installments_payments_outcome_df
gc.collect()
print('after merged installments_payments_outcome_df merged_df.shape is ', merged_df.shape)


previous_application_outcome_df = pd.read_csv('C:/D_Disk/data_competition/home_credit/previous_application_outcome_df_v1.csv',
                                              index_col=0, header=0)
merged_df = merged_df.join(previous_application_outcome_df, how='left')
del previous_application_outcome_df
gc.collect()
print('after merged previous_application_outcome_df merged_df.shape is ', merged_df.shape)

#print(merged_X_joined_df.head())
merged_df = pd.get_dummies(merged_df)
print('after get dummies, merged_df.shape is ', merged_df.shape)
merged_df = merged_df.apply(lambda x: x.fillna(x.mean()),axis=0)
#imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
#imp = imp.fit(merged_df)

merged_X_train = merged_df[merged_df['TARGET']!=-100]
train_y = merged_X_train['TARGET']
#merged_X_train = merged_X_train.drop(['TARGET', 'SK_ID_CURR'], axis=1)
merged_X_train = merged_X_train.drop(['TARGET'], axis=1)

merged_X_test = merged_df[merged_df['TARGET']==-100]
merged_X_test = merged_X_test.drop(['TARGET'], axis=1)

X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(merged_X_train, 
            train_y, test_size=0.2, random_state=42)

print('X_train_new.shape is ', X_train_new.shape)

#-----------------------------------------------------------------------------

print('start training with v1')
rf = RandomForestClassifier(n_estimators=300, n_jobs=2, min_samples_split=50, 
                            max_leaf_nodes=2000, random_state=42, max_depth=37)
start_t = time.time()
rf.fit(X_train_new, y_train_new)
y_predictions_rf = rf.predict_proba(X_val_new)[:,1]
auc_rf = roc_auc_score(y_val_new, y_predictions_rf)
print('partial datat auc_rf: {}, cost time: {}'.format(auc_rf, time.time()-start_t))

merged_X_test['TARGET'] = rf.predict_proba(merged_X_test)[:,1]
merged_X_test['TARGET'].to_csv('C:/D_Disk/data_competition/home_credit/all/outcome_rf_partial1.csv',
          header=['TARGET'])
del rf
gc.collect()

#-----------------------------------------------------------------------------

print('start training with v2')
rf = RandomForestClassifier(n_estimators=1000, n_jobs=2, min_samples_split=50, 
                            max_leaf_nodes=2000, random_state=42, max_depth=33)
start_t = time.time()
rf.fit(X_train_new, y_train_new)
y_predictions_rf = rf.predict_proba(X_val_new)[:,1]
auc_rf = roc_auc_score(y_val_new, y_predictions_rf)
print('partial datat auc_rf: {}, cost time: {}'.format(auc_rf, time.time()-start_t))

merged_X_test = merged_X_test.drop(['TARGET'], axis=1)
merged_X_test['TARGET'] = rf.predict_proba(merged_X_test)[:,1]
merged_X_test['TARGET'].to_csv('C:/D_Disk/data_competition/home_credit/all/outcome_rf_partial2.csv',
          header=['TARGET'])
del rf
gc.collect()

#-----------------------------------------------------------------------------