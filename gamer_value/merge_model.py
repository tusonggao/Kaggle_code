import time
import numpy as np
import pandas as pd

start_t = time.time()
print('start prog')

outcome_df1 = pd.read_csv('C:/D_Disk/data_competition/gamer_value/outcome/submission_full_56.4395_8d9440dac1a25cbd2c61aae4977810cb.csv')

outcome_df2 = pd.read_csv('C:/D_Disk/data_competition/gamer_value/outcome/submission_example_new.csv')

outcome_df1['prediction_pay_price'] += outcome_df2['prediction_pay_price']
outcome_df1['prediction_pay_price'] *= 0.5
outcome_df1['prediction_pay_price'] = outcome_df1['prediction_pay_price'].round(5)

outcome_df1.to_csv('C:/D_Disk/data_competition/gamer_value/outcome/submission_merged_new.csv', index=False)

print('cost time: ', time.time()-start_t)