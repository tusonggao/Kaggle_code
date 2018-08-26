import gc
import time

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

def process_installments_payments():
    # DAYS_ENTRY_PAYMENT.shape is 13605401, 8
    with timer("read installments df"):
        ins = pd.read_csv('C:/D_Disk/data_competition/home_credit/data/'
                          'installments_payments.csv', header=0)
        
    ins, cat_cols = one_hot_encoder(ins, nan_as_category= True)
    print('cat_cols is ', cat_cols)
    
    nan_msk = pd.isna(ins['DAYS_ENTRY_PAYMENT'])
    ins.loc[nan_msk, 'DAYS_ENTRY_PAYMENT'] = (ins.loc[nan_msk, 'DAYS_INSTALMENT'])
    ins.loc[nan_msk, 'AMT_PAYMENT'] = (ins.loc[nan_msk, 'AMT_INSTALMENT'])
        
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)

    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['min', 'max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['min', 'max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['min', 'max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_INSTALMENT': ['min', 'max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
        
    print('start agg')
    with timer('start ins agg'):
        ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    
    print('ins_agg.columns.tolist() is ', ins_agg.columns.tolist())
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper()
                                for e in ins_agg.columns.tolist()])
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    
    ins_agg.to_csv('C:/D_Disk/data_competition/home_credit/data/'
                   'processed_new/installments_payments_outcome_df_new.csv')
    print('ins_agg.columns is ', ins_agg.columns)
    
    del ins, ins_agg
    gc.collect()
    

if __name__=='__main__':
    print('hello world!')
    
    with timer('func process_installments_payments()'):
        process_installments_payments()

    
    