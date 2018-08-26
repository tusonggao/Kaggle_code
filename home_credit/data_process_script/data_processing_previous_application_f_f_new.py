import time
import gc

import numpy as np
import pandas as pd

from contextlib import contextmanager
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))
    
    
# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    print('in one_hot_encoder')
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


def process_previous_application():
    with timer('read previous_application.csv'):
        prev = pd.read_csv('C:/D_Disk/data_competition/home_credit/data/'
                           'previous_application.csv', header=0)
    
    prev, cat_cols = one_hot_encoder(prev, nan_as_category= True)
    print('cat_cols is ', cat_cols)
    
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)
    
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    
    prev['APP_CREDIT_to_ANNUITY_PERC'] = prev['AMT_CREDIT'] / prev['AMT_ANNUITY']
    
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'APP_CREDIT_to_ANNUITY_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']
    
    print('start prev_agg')
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() 
                                 for e in prev_agg.columns.tolist()])
    prev_agg['PREV_APPLICATION_COUNT'] = prev.groupby('SK_ID_CURR').size()
    print('prev_agg finished.')
    
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    cols = approved_agg.columns.tolist()
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() 
                         for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() 
                         for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left')
    
#    del refused, refused_agg, approved, approved_agg, prev
    
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    
    for e in cols:
        prev_agg['NEW_RATIO_PREV_' + e[0] + "_" + e[1].upper()] = (
                prev_agg['APPROVED_' + e[0] + "_" + e[1].upper()] /
                prev_agg['REFUSED_' + e[0] + "_" + e[1].upper()] )
    print('all computation finished')
    with timer('write previous_application_outcome_df.csv'):
        prev_agg.to_csv('C:/D_Disk/data_competition/home_credit/data/'
                    'processed_new/previous_application_outcome_df.csv')
        print('prev_agg.columns is ', prev_agg.columns)
    
    del prev_agg
    gc.collect()


if __name__=='__main__':
    print('hello world!')
    
    with timer('func process_previous_application()'):
        process_previous_application()

    
    