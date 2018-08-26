import numpy as np
import pandas as pd
import time


def process_installments_payments():
    print('in process_installments_payments')
    installments_payments_df = pd.read_csv('C:/D_Disk/data_competition/home_credit/all/installments_payments.csv',
                            header=0)
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
        
        for col_name in ['NUM_INSTALMENT_VERSION', 'NUM_INSTALMENT_NUMBER', 'DAYS_INSTALMENT', 
                         'DAYS_ENTRY_PAYMENT', 'AMT_INSTALMENT', 'AMT_PAYMENT']:
            outcome[col_name + '_avg'] = group[col_name].mean()        
            outcome[col_name + '_sum'] = group[col_name].sum()
            outcome[col_name + '_max'] = group[col_name].max()
            outcome[col_name + '_min'] = group[col_name].min()
            outcome[col_name + '_var'] = group[col_name].var()
            
#        print(name)
#        print(group)
#        print('outcome is ', outcome)
            
        outcome_list.append(outcome)
#        if cnt==8:
        if cnt%3000==0:
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
    installments_payments_outcome_df.to_csv('C:/D_Disk/data_competition/home_credit/installments_payments_outcome_df_new.csv')


def process_previous_application():
    print('in process_previous_application')
    
    previous_application_df = pd.read_csv('C:/D_Disk/data_competition/home_credit/all/previous_application.csv',
                            header=0)
    previous_application_df_by_skid = previous_application_df.groupby('SK_ID_CURR')
    loop_total_num = len(previous_application_df_by_skid)
    
    outcome_list = []
    cnt = 0
    start_t = time.time()
    for name, group in previous_application_df_by_skid:
        outcome = dict()
        cnt += 1
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
            outcome[col_name + '_var'] = group[col_name].var()
            
        for col_name in ['WEEKDAY_APPR_PROCESS_START', 'FLAG_LAST_APPL_PER_CONTRACT',
                         'NFLAG_LAST_APPL_IN_DAY', 'NAME_CASH_LOAN_PURPOSE',
                         'NAME_CONTRACT_STATUS', 'NAME_PAYMENT_TYPE', 'CODE_REJECT_REASON',
                         'NAME_TYPE_SUITE', 'NAME_CLIENT_TYPE', 'NAME_GOODS_CATEGORY',
                         'NAME_PORTFOLIO', 'NAME_PRODUCT_TYPE', 'CHANNEL_TYPE',
                         'NAME_SELLER_INDUSTRY', 'NAME_YIELD_GROUP', 'PRODUCT_COMBINATION',
                         'NFLAG_INSURED_ON_APPROVAL']:
            for indx, val in group[col_name].value_counts().iteritems():
                outcome[col_name + '_num_' + str(indx)] = val
            
        outcome_list.append(outcome)
#        if cnt==8:
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
    previous_application_outcome_df.to_csv('C:/D_Disk/data_competition/home_credit/previous_application_outcome_df_new.csv')

def process_credit_card_balance():
    print('process_credit_card_balance')
    credit_card_balance_df = pd.read_csv('C:/D_Disk/data_competition/home_credit/all/credit_card_balance.csv',
                            header=0)

    credit_card_balance_df.sort_values(by=['MONTHS_BALANCE'], inplace=True)
    credit_card_balance_df_by_skid = credit_card_balance_df.groupby('SK_ID_CURR')
    loop_total_num = len(credit_card_balance_df_by_skid)
    
    outcome_list = []
    cnt = 0
    start_t = time.time()
    for name, group in credit_card_balance_df_by_skid:
        outcome = dict()
        cnt += 1
#        print('cnt is ', cnt)
        outcome['SK_ID_CURR'] = name
        outcome['record_num'] = group.shape[0]
        
        for col_name in ['MONTHS_BALANCE', 'AMT_BALANCE', 'AMT_CREDIT_LIMIT_ACTUAL',
                         'AMT_DRAWINGS_ATM_CURRENT', 'AMT_DRAWINGS_OTHER_CURRENT',
                         'AMT_DRAWINGS_POS_CURRENT', 'AMT_INST_MIN_REGULARITY',
                         'AMT_PAYMENT_TOTAL_CURRENT', 'AMT_RECEIVABLE_PRINCIPAL',
                         'AMT_RECIVABLE', 'AMT_TOTAL_RECEIVABLE',
                         'CNT_DRAWINGS_ATM_CURRENT', 'CNT_DRAWINGS_CURRENT',
                         'CNT_DRAWINGS_OTHER_CURRENT', 'CNT_DRAWINGS_POS_CURRENT',
                         'CNT_INSTALMENT_MATURE_CUM', 'SK_DPD', 'SK_DPD_DEF']:
            outcome[col_name + '_avg'] = group[col_name].mean()        
            outcome[col_name + '_sum'] = group[col_name].sum()
            outcome[col_name + '_max'] = group[col_name].max()
            outcome[col_name + '_min'] = group[col_name].min()
            outcome[col_name + '_var'] = group[col_name].var()
        
        outcome['SK_DPD_unique_num'] = len(group['SK_DPD'].unique())
        outcome['SK_DPD_last'] = group['SK_DPD'].iloc[-1]
            
        outcome['SK_DPD_DEF_unique_num'] = len(group['SK_DPD_DEF'].unique())
        outcome['SK_DPD_last'] = group['SK_DPD_DEF'].iloc[-1]        
        
#        print(name)
#        print(group)
#        print('outcome is ', outcome)
            
        outcome_list.append(outcome)
#        if cnt==8:
        if cnt%3000==0:
            end_t = time.time()
            per_loop_t = (end_t-start_t)/cnt
            time_left = per_loop_t*(loop_total_num-cnt)
            print('cnt is {}, average time per loop is {}, time left is {}'.format(
                    cnt, per_loop_t, time_left))
#            break

    credit_card_balance_outcome_df = pd.DataFrame(outcome_list)
    credit_card_balance_outcome_df.set_index('SK_ID_CURR', inplace=True)    
    credit_card_balance_outcome_df.rename(columns=lambda x: 'credit_card_balance_' + x, inplace=True)    
    print('credit_card_balance_outcome_df.shape is ', credit_card_balance_outcome_df.shape)
    credit_card_balance_outcome_df.to_csv('C:/D_Disk/data_competition/home_credit/credit_card_balance_outcome_df_new.csv')


def process_bureau():
    print('in process_bureau')
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
            outcome[col_name + '_var'] = group[col_name].var()
            
        for col_name in ['CNT_CREDIT_PROLONG', 'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT',
                         'AMT_CREDIT_SUM_LIMIT', 'AMT_CREDIT_SUM_OVERDUE',
                         'DAYS_CREDIT_UPDATE', 'AMT_ANNUITY', 'DAYS_CREDIT']:
            outcome[col_name + '_avg'] = group[col_name].mean()        
            outcome[col_name + '_sum'] = group[col_name].sum()
            outcome[col_name + '_max'] = group[col_name].max()
            outcome[col_name + '_min'] = group[col_name].min()
            outcome[col_name + '_var'] = group[col_name].var()
            
        
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
    bureau_outcome_df.to_csv('C:/D_Disk/data_competition/home_credit/bureau_outcome_df_new.csv')
        
    
def process_bureau_balance():
    print('in process_bureau_balance')
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
        outcome['MONTHS_BALANCE_var'] = group['MONTHS_BALANCE'].var()
        
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
    bureau_outcome_df.to_csv('C:/D_Disk/data_competition/home_credit/bureau_balance_outcome_df_new.csv')






if __name__=='__main__':
    print('hello world!')
    
    process_previous_application()
    process_installments_payments()
    process_credit_card_balance()
    process_bureau()
    process_bureau_balance()

    
    