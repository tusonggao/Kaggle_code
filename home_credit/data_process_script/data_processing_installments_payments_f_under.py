import gc

import numpy as np
import pandas as pd
import time

def process_installments_payments():
    # DAYS_ENTRY_PAYMENT.shape is 13605401, 8
    installments_payments_df = pd.read_csv('C:/D_Disk/data_competition/home_credit/data/installments_payments.csv',
                            header=0)
    
    nan_msk = pd.isna(installments_payments_df['DAYS_ENTRY_PAYMENT'])
    installments_payments_df.loc[nan_msk, 'DAYS_ENTRY_PAYMENT'] = (
            installments_payments_df.loc[nan_msk, 'DAYS_INSTALMENT'])
    installments_payments_df.loc[nan_msk, 'AMT_PAYMENT'] = (
            installments_payments_df.loc[nan_msk, 'AMT_INSTALMENT'])
        
    
    installments_payments_df_by_skid = installments_payments_df.groupby('SK_ID_CURR')
    loop_total_num = len(installments_payments_df_by_skid)
    
    outcome_list = []
    cnt = 0
    start_t = time.time()
    for name, group in installments_payments_df_by_skid:
        outcome = dict()
        cnt += 1
#        print('cnt is ', cnt)
        outcome['SK_ID_CURR'] = name
        outcome['record_num'] = group.shape[0]
        
        outcome['NUM_INSTALMENT_NUMBER_max'] = np.max
        
        # NUM_INSTALMENT_VERSION， NUM_INSTALMENT_VERSION DAYS_INSTALMENT AMT_INSTALMENT 没有空值
        # DAYS_ENTRY_PAYMENT AMT_PAYMENT 都只有 2905个空值
        
        for col_name in ['NUM_INSTALMENT_VERSION', 'NUM_INSTALMENT_NUMBER', 'DAYS_INSTALMENT', 
                         'DAYS_ENTRY_PAYMENT', 'AMT_INSTALMENT', 'AMT_PAYMENT']:
            outcome[col_name + '_avg'] = np.mean(group[col_name])  
            outcome[col_name + '_median'] = np.median(group[col_name])  
            outcome[col_name + '_sum'] = np.sum(group[col_name])
            outcome[col_name + '_max'] = np.max(group[col_name])
            outcome[col_name + '_min'] = np.min(group[col_name])
            outcome[col_name + '_var'] = np.var(group[col_name])
            outcome[col_name + '_std'] = np.std(group[col_name])
            
            outcome[col_name + '_nan_avg'] = np.nanmean(group[col_name])  
            outcome[col_name + '_nan_median'] = np.nanmedian(group[col_name]) 
            outcome[col_name + '_nan_sum'] = np.nansum(group[col_name])
            outcome[col_name + '_nan_max'] = np.nanmax(group[col_name])
            outcome[col_name + '_nan_min'] = np.nanmin(group[col_name])
            outcome[col_name + '_nan_var'] = np.nanvar(group[col_name])
            outcome[col_name + '_nan_std'] = np.nanstd(group[col_name])
            
#        print(name)
#        print(group)
#        print('outcome is ', outcome)
            
        outcome_list.append(outcome)
        
#        if cnt==8:
#            break
        
        if cnt%5000==0:
            end_t = time.time()
            per_loop_t = (end_t-start_t)/cnt
            time_left = per_loop_t*(loop_total_num-cnt)
            print('loop_total_num is {}, cnt is {}, average time per loop is {}, time left is {}'.format(
                 loop_total_num, cnt, per_loop_t, time_left))
#            break

    installments_payments_outcome_df = pd.DataFrame(outcome_list)
    installments_payments_outcome_df.set_index('SK_ID_CURR', inplace=True)    
    installments_payments_outcome_df.rename(columns=lambda x: 'installments_payments_' + x, inplace=True)    
    print('installments_payments_outcome_df.shape is ', installments_payments_outcome_df.shape)
    installments_payments_outcome_df.to_csv('C:/D_Disk/data_competition'
           '/home_credit/data/processed_new/installments_payments_outcome_df.csv')

    del installments_payments_outcome_df, outcome_list
    gc.collect()
    

if __name__=='__main__':
    print('hello world!')
    
    process_installments_payments()

    
    