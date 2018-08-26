import pickle
import time
import gc
import os

import pandas as pd
import numpy as np

data_dir_path = 'C:/D_Disk/data_competition/home_credit/data/'

train_df = pd.read_csv(data_dir_path + 'application_train.csv', index_col=0, header=0)
test_df = pd.read_csv(data_dir_path + 'application_test.csv', index_col=0, header=0)
test_df['TARGET'] = np.nan

merged_df = train_df.append(test_df)
print('original merged_df.shape is ', merged_df.shape)

###########################################################################

# some data cleanning
#merged_df = merged_df[merged_df['CODE_GENDER'] != 'XNA']

merged_df['CODE_GENDER'].replace('XNA', np.nan, inplace=True)
merged_df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
merged_df['DAYS_LAST_PHONE_CHANGE'].replace(0, np.nan, inplace=True)
merged_df['NAME_FAMILY_STATUS'].replace('Unknown', np.nan, inplace=True)
merged_df['ORGANIZATION_TYPE'].replace('XNA', np.nan, inplace=True)

###########################################################################

# Some simple new features (percentages)
#merged_df['DAYS_EMPLOYED_PERC'] = merged_df['DAYS_EMPLOYED'] / merged_df['DAYS_BIRTH']
#merged_df['INCOME_CREDIT_PERC'] = merged_df['AMT_INCOME_TOTAL'] / merged_df['AMT_CREDIT']
#merged_df['INCOME_PER_PERSON'] = merged_df['AMT_INCOME_TOTAL'] / merged_df['CNT_FAM_MEMBERS']
#merged_df['ANNUITY_INCOME_PERC'] = merged_df['AMT_ANNUITY'] / merged_df['AMT_INCOME_TOTAL']
#merged_df['PAYMENT_RATE'] = merged_df['AMT_ANNUITY'] / merged_df['AMT_CREDIT']

doc_flag_cols = [col for col in merged_df.columns if 'FLAG_DOC' in col]
merged_df['doc_flag_num_sum'] = merged_df[doc_flag_cols].sum(axis=1)

non_doc_flag_cols = [col for col in merged_df.columns if 
                     ('FLAG_' in col) and ('FLAG_DOC' not in col)]
merged_df['non_doc_flag_num_sum'] = merged_df[doc_flag_cols].sum(axis=1)

merged_df['annuity_income_percentage'] = merged_df['AMT_ANNUITY'] / merged_df['AMT_INCOME_TOTAL']
merged_df['car_to_birth_ratio'] = merged_df['OWN_CAR_AGE'] / merged_df['DAYS_BIRTH']
merged_df['car_to_employ_ratio'] = merged_df['OWN_CAR_AGE'] / merged_df['DAYS_EMPLOYED']
merged_df['children_ratio'] = merged_df['CNT_CHILDREN'] / merged_df['CNT_FAM_MEMBERS']
merged_df['credit_to_annuity_ratio'] = merged_df['AMT_CREDIT'] / merged_df['AMT_ANNUITY']
merged_df['credit_to_goods_ratio'] = merged_df['AMT_CREDIT'] / merged_df['AMT_GOODS_PRICE']
merged_df['credit_to_income_ratio'] = merged_df['AMT_CREDIT'] / merged_df['AMT_INCOME_TOTAL']
merged_df['days_employed_percentage'] = merged_df['DAYS_EMPLOYED'] / merged_df['DAYS_BIRTH']
merged_df['income_credit_percentage'] = merged_df['AMT_INCOME_TOTAL'] / merged_df['AMT_CREDIT']
merged_df['income_per_child'] = merged_df['AMT_INCOME_TOTAL'] / (1 + merged_df['CNT_CHILDREN'])
merged_df['income_per_person'] = merged_df['AMT_INCOME_TOTAL'] / merged_df['CNT_FAM_MEMBERS']
merged_df['payment_rate'] = merged_df['AMT_ANNUITY'] / merged_df['AMT_CREDIT']
merged_df['phone_to_birth_ratio'] = merged_df['DAYS_LAST_PHONE_CHANGE'] / merged_df['DAYS_BIRTH']
merged_df['phone_to_employ_ratio'] = merged_df['DAYS_LAST_PHONE_CHANGE'] / merged_df['DAYS_EMPLOYED']

merged_df['cnt_non_child'] = merged_df['CNT_FAM_MEMBERS'] - merged_df['CNT_CHILDREN']
merged_df['child_to_non_child_ratio'] = merged_df['CNT_CHILDREN'] / merged_df['cnt_non_child']
merged_df['income_per_non_child'] = merged_df['AMT_INCOME_TOTAL'] / merged_df['cnt_non_child']
merged_df['credit_per_person'] = merged_df['AMT_CREDIT'] / merged_df['CNT_FAM_MEMBERS']
merged_df['credit_per_child'] = merged_df['AMT_CREDIT'] / (1 + merged_df['CNT_CHILDREN'])
merged_df['credit_per_non_child'] = merged_df['AMT_CREDIT'] / merged_df['cnt_non_child']             

merged_df['external_sources_weighted'] = (merged_df.EXT_SOURCE_1 * 2 + 
                                          merged_df.EXT_SOURCE_2 * 3 + 
                                          merged_df.EXT_SOURCE_3 * 4)
for function_name in ['min', 'nanmin', 'max', 'nanmax',  'sum', 'nansum', 
                      'mean', 'nanmean', 'median', 'nanmedian']:
    merged_df['external_sources_{}'.format(function_name)] = eval('np.{}'.format(function_name))(
        merged_df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']], axis=1)

merged_df['short_employment'] = (merged_df['DAYS_EMPLOYED'] < -2000).astype(int)
merged_df['young_age'] = (merged_df['DAYS_BIRTH'] < -14000).astype(int)

print('after cleanning and adding new featuers, merged_df.shape is ',
      merged_df.shape)

###########################################################################

pos_balance_df = pd.read_csv(data_dir_path + 
                             '/processed_new/pos_cash_balance_outcome_df.csv',
                             index_col=0, header=0)
pos_balance_df.rename(columns=lambda x: 'pos_balance_' + x, inplace=True)
merged_df = merged_df.join(pos_balance_df, how='left')
del pos_balance_df
gc.collect()
print('after merged pos_balance_df merged_df.shape is ', merged_df.shape)

#############################################################################

bureau_outcome_df = pd.read_csv(data_dir_path + '/processed_new/bureau_outcome_df.csv', 
                                index_col=0, header=0)
bureau_outcome_df.rename(columns=lambda x: 'bureau_outcome_' + x, inplace=True)    
merged_df = merged_df.join(bureau_outcome_df, how='left')
del bureau_outcome_df
gc.collect()
print('after merged bureau_outcome_df merged_df.shape is ', merged_df.shape)

#################################################################################

credit_card_balance_outcome_df = pd.read_csv(data_dir_path + 
            '/processed_new/credit_card_balance_outcome_df_new.csv',
            index_col=0, header=0)
merged_df = merged_df.join(credit_card_balance_outcome_df, how='left')
del credit_card_balance_outcome_df
gc.collect()
print('after merged credit_card_balance_outcome_df merged_df.shape is ', merged_df.shape)


#################################################################################

installments_payments_outcome_df = pd.read_csv(data_dir_path + 
        '/processed_new/installments_payments_outcome_df_new.csv', 
        index_col=0, header=0)
merged_df = merged_df.join(installments_payments_outcome_df, how='left')
del installments_payments_outcome_df
gc.collect()
print('after merged installments_payments_outcome_df merged_df.shape is ', 
      merged_df.shape)


#################################################################################

previous_application_outcome_df = pd.read_csv(data_dir_path + 
                  '/processed_new/previous_application_outcome_df.csv',
                  index_col=0, header=0)
merged_df['has_previous_application'] = 0
merged_df.loc[previous_application_outcome_df.index, 'has_previous_application'] = 1
merged_df = merged_df.join(previous_application_outcome_df, how='left')

del previous_application_outcome_df
gc.collect()
print('after merged previous_application_outcome_df merged_df.shape is ', 
      merged_df.shape)

#################################################################################

print('final merged_df.shape is ', merged_df.shape)

#start_t = time.time()
#merged_df.to_csv(data_dir_path + '/processed/merged_df_noround.csv')
#print('store to file merged_df_noround.csv cost time:', time.time()-start_t)

start_t = time.time()
merged_df = merged_df.round(2)
merged_df.to_csv(data_dir_path + '/processed/merged_df.csv')
print('store to file merged_df.csv cost time:', time.time()-start_t)

#merged_df = pd.get_dummies(merged_df)
#print('after get dummies, merged_df.shape is ', merged_df.shape)