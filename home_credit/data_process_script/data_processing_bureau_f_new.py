import time
import gc

import numpy as np
import pandas as pd

from contextlib import contextmanager
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

data_path = 'C:/D_Disk/data_competition/home_credit/data/'

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category=True):
    print('in one_hot_encoder')
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
#    print('categorical_columns is ', categorical_columns)
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
#    print('new_columns is ', new_columns)
    return df, new_columns

def process_bureau():
    with timer('read bureau.csv, bureau_balance.csv'):
        bureau = pd.read_csv(data_path + 'bureau.csv', header=0)
        bb = pd.read_csv(data_path + 'bureau_balance.csv', header=0)
    
    bb, bb_cat = one_hot_encoder(bb, nan_as_category=False)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category=False)
    
    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper()
                               for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    
    del bb, bb_agg
    gc.collect()
    
    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']
    
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper()
                                   for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    bureau_agg['POS_COUNT'] = bureau.groupby('SK_ID_CURR').size()
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    cols = active_agg.columns.tolist()
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper()
                                   for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left')
    
    del active, active_agg
    gc.collect()
    
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper()
                                   for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left')
    
    for e in cols:
        bureau_agg['NEW_RATIO_BURO_' + e[0] + "_" + e[1].upper()] = (
                bureau_agg['ACTIVE_' + e[0] + "_" + e[1].upper()] / 
                bureau_agg['CLOSED_' + e[0] + "_" + e[1].upper()] )
    
    print('computation finished, bureau_agg.shape is ', bureau_agg.shape)
    with timer('write bureau_outcome_df.csv'):
        bureau_agg.to_csv(data_path + '/processed_new/bureau_outcome_df.csv')
    
    del closed, closed_agg, bureau, bureau_agg
    gc.collect()

def pos_cash():
    pos = pd.read_csv(data_path + 'POS_CASH_balance.csv')
    pos, cat_cols = one_hot_encoder(pos, nan_as_category= False)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper()
                                for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    print('computation finished, pos_agg.shape is', pos_agg.shape)
    
    with timer('write pos_cash_balance_outcome_df.csv'):
        pos_agg.to_csv(data_path + '/processed_new/pos_cash_balance_outcome_df.csv')
    
    del pos, pos_agg
    gc.collect()

if __name__=='__main__':
    print('hello world!')
    
    with timer('func process_bureau'):
        process_bureau()
        
    with timer('func pos_cash'):
        pos_cash()

    
    