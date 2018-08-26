import numpy as np
import pandas as pd
import time

def process_bureau():
    bureau_df = pd.read_csv('C:/D_Disk/data_competition/home_credit/all/bureau.csv',
                            header=0)
    
    bureau_balance_outcome_df = pd.read_csv('C:/D_Disk/data_competition/home_credit/bureau_balance_outcome_df.csv',
                            header=0)
    
    
    bureau_df = pd.merge(bureau_df, bureau_balance_outcome_df, how='left', on='SK_ID_BUREAU')

    
#    bureau_joined_df = pd.merge(bureau_df, bureau_balance_df, on='SK_ID_BUREAU')
#    print(bureau_joined_df.head())
#    bureau_df_by_skid = bureau_joined_df.groupby('SK_ID_CURR')
    
    bureau_df.sort_values(by=['DAYS_CREDIT'], inplace=True)
    bureau_df_by_skid = bureau_df.groupby('SK_ID_CURR')
    loop_total_num = len(bureau_df_by_skid)
    
    outcome_list = []
    cnt = 0
    start_t = time.time()
    for name, group in bureau_df_by_skid:
        outcome = dict()
        cnt += 1
#        print('cnt is ', cnt)
        outcome['SK_ID_CURR'] = name
        outcome['bureau_record_num'] = group.shape[0]
        
        for col_name in ['bureau_balance_MONTHS_BALANCE_avg', 'bureau_balance_MONTHS_BALANCE_max', 
                         'bureau_balance_MONTHS_BALANCE_min', 'bureau_balance_MONTHS_BALANCE_sum', 
                         'bureau_balance_record_num',   'bureau_balance_STATUS_num_0', 
                         'bureau_balance_STATUS_num_1', 'bureau_balance_STATUS_num_2', 
                         'bureau_balance_STATUS_num_3', 'bureau_balance_STATUS_num_4', 
                         'bureau_balance_STATUS_num_5', 'bureau_balance_STATUS_num_C', 
                         'bureau_balance_STATUS_num_X', 'bureau_balance_STATUS_unique_num']:
            outcome[col_name + '_avg'] = group[col_name].mean()        
            outcome[col_name + '_sum'] = group[col_name].sum()
            outcome[col_name + '_max'] = group[col_name].max()
            outcome[col_name + '_min'] = group[col_name].min()
            
        for col_name in ['CNT_CREDIT_PROLONG', 'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT',
                         'AMT_CREDIT_SUM_LIMIT', 'AMT_CREDIT_SUM_OVERDUE',
                         'DAYS_CREDIT_UPDATE', 'AMT_ANNUITY', 'DAYS_CREDIT']:
            outcome[col_name + '_avg'] = group[col_name].mean()        
            outcome[col_name + '_sum'] = group[col_name].sum()
            outcome[col_name + '_max'] = group[col_name].max()
            outcome[col_name + '_min'] = group[col_name].min()
            
        
        outcome['CREDIT_CURRENCY_unique_num'] = len(group['CREDIT_CURRENCY'].unique())
        outcome['CREDIT_CURRENCY_last'] = group['CREDIT_CURRENCY'].iloc[-1]
        for indx, val in group['CREDIT_CURRENCY'].value_counts().iteritems():
            outcome['CREDIT_CURRENCY_num_' + indx] = val
        
        outcome['CREDIT_TYPE_unique_num'] = len(group['CREDIT_TYPE'].unique())
        outcome['CREDIT_TYPE_last'] = group['CREDIT_TYPE'].iloc[-1]
        for indx, val in group['CREDIT_TYPE'].value_counts().iteritems():
            outcome['CREDIT_TYPE_num_' + indx] = val
        
        outcome['CREDIT_ACTIVE_last'] = group['CREDIT_ACTIVE'].iloc[-1]
        for indx, val in group['CREDIT_ACTIVE'].value_counts().iteritems():
            outcome['CREDIT_ACTIVE_num_' + indx] = val
            
#        print(name)
#        print(group)
#        print('outcome is ', outcome)
            
        outcome_list.append(outcome)
#        if cnt==8:
        if cnt%1000==0:
            end_t = time.time()
            per_loop_t = (end_t-start_t)/cnt
            time_left = per_loop_t*(loop_total_num-cnt)
            print('cnt is {}, average time per loop is {}, time left is {}'.format(
                    cnt, per_loop_t, time_left))

    bureau_outcome_df = pd.DataFrame(outcome_list)
    bureau_outcome_df.set_index('SK_ID_CURR', inplace=True)    
    bureau_outcome_df.rename(columns=lambda x: 'bureau_' + x, inplace=True)    
    print('bureau_outcome_df.shape is ', bureau_outcome_df.shape)
    bureau_outcome_df.to_csv('C:/D_Disk/data_competition/home_credit/bureau_outcome_df_v3.csv')
        
    
def process_bureau_balance():
    bureau_balance_df = pd.read_csv('C:/D_Disk/data_competition/home_credit/all/bureau_balance.csv',
                                    header=0)    

    bureau_balance_df_by_skid = bureau_balance_df.groupby('SK_ID_BUREAU')
    loop_total_num = len(bureau_balance_df_by_skid)
    
    outcome_list = []
    cnt = 0
    start_t = time.time()
    for name, group in bureau_balance_df_by_skid:
        outcome = dict()
        cnt += 1
        outcome['SK_ID_BUREAU'] = name
        outcome['record_num'] = group.shape[0]
        
        outcome['MONTHS_BALANCE_avg'] = group['MONTHS_BALANCE'].mean()        
        outcome['MONTHS_BALANCE_sum'] = group['MONTHS_BALANCE'].sum()
        outcome['MONTHS_BALANCE_max'] = group['MONTHS_BALANCE'].max()
        outcome['MONTHS_BALANCE_min'] = group['MONTHS_BALANCE'].min()
        
        outcome['STATUS_unique_num'] = len(group['STATUS'].unique())
        outcome['STATUS_last'] = group['STATUS'].iloc[-1]
        for indx, val in group['STATUS'].value_counts().iteritems():
            outcome['STATUS_num_' + indx] = val
            outcome['STATUS_ratio_' + indx] = val/group.shape[0]
            
#        print(name)
#        print(group)
#        print('outcome is ', outcome)
        outcome_list.append(outcome)
        
#        if cnt<=10:
        if cnt%1000==0:
            end_t = time.time()
            per_loop_t = (end_t-start_t)/cnt
            time_left = per_loop_t*(loop_total_num-cnt)
            print('cnt is {}, average time per loop is {}, time left is {}'.format(
                    cnt, per_loop_t, time_left))
            

    bureau_outcome_df = pd.DataFrame(outcome_list)
    bureau_outcome_df.set_index('SK_ID_BUREAU', inplace=True)    
    bureau_outcome_df.rename(columns=lambda x: 'bureau_balance_' + x, inplace=True)    
    bureau_outcome_df.to_csv('C:/D_Disk/data_competition/home_credit/bureau_balance_outcome_df.csv')
        


if __name__=='__main__':
    print('hello world!')
    
    process_bureau()
#    process_bureau_balance()

    
    