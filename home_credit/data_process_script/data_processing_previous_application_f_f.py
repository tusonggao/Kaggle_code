import numpy as np
import pandas as pd
import time

def process_previous_application():
    previous_application_df = pd.read_csv('C:/D_Disk/data_competition/'
                  'home_credit/data/previous_application.csv', header=0)
    
    previous_application_df['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    previous_application_df['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    previous_application_df['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    previous_application_df['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    previous_application_df['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)
        
    previous_application_df_by_skid = previous_application_df.groupby('SK_ID_CURR')
    loop_total_num = len(previous_application_df_by_skid)
    
    outcome_list = []
    cnt = 0
    start_t = time.time()
    for name, group in previous_application_df_by_skid:
        outcome = dict()
        cnt += 1
#        print('cnt is ', cnt)
        outcome['SK_ID_CURR'] = name
        outcome['record_num'] = group.shape[0]
        
        for col_name in ['AMT_ANNUITY', 'AMT_APPLICATION', 'AMT_CREDIT', 
                         'AMT_DOWN_PAYMENT', 'AMT_GOODS_PRICE',
                         'HOUR_APPR_PROCESS_START', 'RATE_DOWN_PAYMENT', 
                         'RATE_INTEREST_PRIMARY', 'RATE_INTEREST_PRIVILEGED',
                         'DAYS_DECISION', 'SELLERPLACE_AREA', 'CNT_PAYMENT',
                         'DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE',
                         'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE',
                         'DAYS_TERMINATION']:
            outcome[col_name + '_avg'] = group[col_name].mean()        
            outcome[col_name + '_sum'] = group[col_name].sum()
            outcome[col_name + '_max'] = group[col_name].max()
            outcome[col_name + '_min'] = group[col_name].min()
            outcome[col_name + '_std'] = group[col_name].std()
            
            outcome[col_name + '_nan_avg'] = np.nanmean(group[col_name])      
            outcome[col_name + '_nan_sum'] = np.nansum(group[col_name])
            outcome[col_name + '_nan_max'] = np.nanmax(group[col_name])
            outcome[col_name + '_nan_min'] = np.nanmin(group[col_name])
            outcome[col_name + '_nan_std'] = np.nanstd(group[col_name])
            
            
        for col_name in ['WEEKDAY_APPR_PROCESS_START', 'FLAG_LAST_APPL_PER_CONTRACT',
                         'NFLAG_LAST_APPL_IN_DAY', 'NAME_CASH_LOAN_PURPOSE',
                         'NAME_CONTRACT_STATUS', 'NAME_PAYMENT_TYPE', 'CODE_REJECT_REASON',
                         'NAME_TYPE_SUITE', 'NAME_CLIENT_TYPE', 'NAME_GOODS_CATEGORY',
                         'NAME_PORTFOLIO', 'NAME_PRODUCT_TYPE', 'CHANNEL_TYPE',
                         'NAME_SELLER_INDUSTRY', 'NAME_YIELD_GROUP', 'PRODUCT_COMBINATION',
                         'NFLAG_INSURED_ON_APPROVAL']:
            for indx, val in group[col_name].value_counts().iteritems():
                outcome[col_name + '_num_' + str(indx)] = val
        
#        print(name)
#        print(group)
#        print('outcome is ', outcome)
            
        outcome_list.append(outcome)
#        if cnt==8:
#            break
        
        if cnt%3000==0:
            end_t = time.time()
            per_loop_t = (end_t-start_t)/cnt
            time_left = per_loop_t*(loop_total_num-cnt)
            print('cnt is {}, average time per loop is {}, time left is {}'.format(
                    cnt, per_loop_t, time_left))
#            break

    previous_application_outcome_df = pd.DataFrame(outcome_list)
    previous_application_outcome_df.set_index('SK_ID_CURR', inplace=True)    
    previous_application_outcome_df.rename(columns=lambda x: 'previous_application_' + x, inplace=True)    
    print('previous_application_outcome_df.shape is ', previous_application_outcome_df.shape)
    previous_application_outcome_df.to_csv('C:/D_Disk/data_competition/'
       'home_credit/data/processed_new/previous_application_outcome_df.csv')
    

if __name__=='__main__':
    print('hello world!')
    
    process_previous_application()

    
    