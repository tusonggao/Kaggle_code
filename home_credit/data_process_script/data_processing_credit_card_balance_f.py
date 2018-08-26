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
    print('categorical_columns is ', categorical_columns)
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    print('new_columns is ', new_columns)
    return df, new_columns

def process_credit_card_balance():
    data_path = 'C:/D_Disk/data_competition/home_credit/data/'
    
    with timer('read credit_card_balance.csv'):
        cc = pd.read_csv(data_path + 'credit_card_balance.csv', header=0)

    cc, cat_cols = one_hot_encoder(cc, nan_as_category=False)
#    print('cat_cols is ', cat_cols)
    
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    
    print('computation finished, cc_agg.shape is ', cc_agg.shape)
#    print('cc_agg columns is ', cc_agg.columns)
    
    with timer('write credit_card_balance_outcome_df_new.csv'):
        cc_agg.to_csv(data_path + '/processed_new/credit_card_balance_outcome_df_new.csv')
    
    del cc, cc_agg
    gc.collect()

if __name__=='__main__':
    print('hello world!')
    
    with timer('func process_credit_card_balance'):
        process_credit_card_balance()

    
    