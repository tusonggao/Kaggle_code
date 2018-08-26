import numpy as np
import pandas as pd

def process_bureau():
    bureau_df = pd.read_csv('C:/D_Disk/data_competition/home_credit/all/bureau.csv',
                            header=0)
    bureau_balance_df = pd.read_csv('C:/D_Disk/data_competition/home_credit/all/bureau_balance.csv',
                                    header=0)
    
    bureau_joined_df = pd.merge(bureau_df, bureau_balance_df, on='SK_ID_BUREAU')
    print(bureau_joined_df.head())
    
    bureau_df_by_skid = bureau_joined_df.groupby('SK_ID_CURR')
    bureau_df_by_skid.size()
    
    print()
    
#    cnt = 0
#    for name, group in bureau_df_by_skid:
#        cnt += 1
#        print(name)
#        print(group)
#        if cnt==3:
#            break
        
        
def process_POS_CASH_balance():
    POS_CASH_balance_df = pd.read_csv('C:/D_Disk/data_competition/home_credit/all/POS_CASH_balance.csv',
                            header=0)    
    POS_CASH_balance_df_by_skid = POS_CASH_balance_df.groupby('SK_ID_CURR')
    POS_CASH_balance_df_by_skid.size()    
    outcome_list = []
    cnt = 0
    for name, group in POS_CASH_balance_df_by_skid:
        outcome = dict()
        cnt += 1
        group = group.sort_values(by=['MONTHS_BALANCE'])
        outcome['SK_ID_CURR'] = name
        outcome['SK_ID_PREV_num'] = group['SK_ID_PREV'].unique().size
        outcome['POS_CASH_record_num'] = group.shape[0]
        outcome['CNT_INSTALMENT_avg'] = group['CNT_INSTALMENT'].mean()
        outcome['CNT_INSTALMENT_FUTURE_avg'] = group['CNT_INSTALMENT_FUTURE'].mean()
        outcome['CNT_INSTALMENT_FUTURE_last'] = group['CNT_INSTALMENT_FUTURE'].iloc[-1]
        outcome['POS_CASH_final_status'] = group['NAME_CONTRACT_STATUS'].iloc[-1]
        
        for contract_status in ['Active', 'Completed', 'Signed', 'Demand', 
                                     'Returned to the store', 'Approved', 
                                     'Amortized debt', 'Canceled', 'XNA']:
            outcome['POS_CASH_status_' + contract_status + '_num'] = np.sum(
                    group['NAME_CONTRACT_STATUS']==contract_status)
#        print(name)
#        print(group)
#        print('outcome is ', outcome)
        
        outcome_list.append(outcome)
#        if cnt==5:
        if cnt%300==0:
            print('cnt is ', cnt)
    outcome_df = pd.DataFrame(outcome_list)
    outcome_df.set_index('SK_ID_CURR', inplace=True)
    outcome_df.to_csv('outcome_df.csv')
#    print(bureau_df_by_skid.size())
#    print(POS_CASH_balance_df['SK_ID_PREV'].unique().shape)
#    print(POS_CASH_balance_df['SK_ID_PREV'].shape)
#    print(POS_CASH_balance_df['SK_ID_PREV'].value_counts())




if __name__=='__main__':
    print('hello world!')
#    train_df = pd.read_csv('C:/D_Disk/data_competition/home_credit/all/application_train.csv', header=0)
#    test_df = pd.read_csv('C:/D_Disk/data_competition/home_credit/all/application_test.csv', header=0)
#
#train_y = train_df['TARGET']
#train_X = train_df.drop(['TARGET'], axis=1)
#test_X = test_df
#merged_X = train_X.append(test_X)
    
#    process_bureau()
    
#    process_POS_CASH_balance()
    process_POS_CASH_balance()
    
    
    
    
    