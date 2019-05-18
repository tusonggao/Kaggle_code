import numpy as np
import pandas as pd
import math
import random
import time

def generate_random_submission():
    print('ha ha ha')
    # 纯随机  36.5563
    # 注意顺序值
    file_w = open('F:/github_me_repos/Kaggle_code/tencent_algo_2019/submission/submission.csv', 'w')
    with open('F:/github_me_repos/Kaggle_code/tencent_algo_2019/submission/test_sample.dat') as file:
        cnt = 0
        for line in file:
            sample_id = line.strip('\n').split('\t')[0]
            time_str = line.strip('\n').split('\t')[2]
            print('time_str is ', time_str)
            # time.strftime("%Y-%m-%d %H:%M:%S", int(time.localtime())
            real_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time_str)))
            print('real_time_str is ', real_time_str)
            exposure_num = random.randint(100, 200)
            file_w.write(sample_id + ',' + str(exposure_num) + '\n')
            cnt += 1
            if cnt >= 3:
                break
    file_w.close()
    return


def SMAPE(evaluate_lst, actual_lst):
    sum = 0.0
    for i in range(len(evaluate_lst)):
        sum += math.fabs(evaluate_lst[i]-actual_lst[i]) / (evaluate_lst[i]+actual_lst[i])
    sum /= len(evaluate_lst)
    return sum

generate_random_submission()