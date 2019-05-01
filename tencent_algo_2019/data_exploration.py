import time
import os
import numpy as np
import pandas as pd

exposure_file = './data/algo.qq.com_641013010_testa/testA/imps_log/totalExposureLog.out'
# 10 列 102386695 行
# 广告请求ID有82085734个不同的  广告位id有329个不同的   用户ID有1341958不同的
# 曝光广告有509280个不同的   曝光广告素材尺寸64个 曝光广告出价bid 19036个不同值
# 曝光广告pctr 有个不同的值
# 广告请求ID  广告请求时间  广告位id  用户ID  曝光广告  曝光广告素材尺寸  曝光广告出价bid  曝光广告pctr  曝光广告quality_ecpm  曝光广告totalEcpm
# line_cnt is  100000 content:  76535481	1550408116	39	926292	118401	36	45	18.812	376.24	1222.24
# line_cnt is  200000 content:  45340639	1550368494	25	660821	206786	30	11	13.492	269.84	418.252
# line_cnt is  300000 content:  29906070	1550333927	209	447125	476095	64	27	8.431	168.62	396.257


user_file = './data/algo.qq.com_641013010_testa/testA/user/user_data'
# 11 列 1396718 行  每一行的user_id都是不同的
# 用户ID  年龄  性别  地域  婚恋状态  学历  消费能力  设备  工作状态  连接类型  行为兴趣
# 用户ID
# line_cnt is  1000000 content:  884336	819	3	-1,-1,1231,12048,11524,-1	0	6	1	3	0	2	0


ad_static_feature_file = './data/algo.qq.com_641013010_testa/testA/ad_static_feature.out'

ad_operation_file = './data/algo.qq.com_641013010_testa/testA/ad_operation.dat'


start_t = time.time()
print('start prog')
with open(exposure_file) as file:
    cnt = 0
    user_id_set = set()
    for line in file:
        cnt += 1
        user_id = line.split('\t')[7]
        user_id_set.add(user_id)
        if cnt%100000==0:
            print('line_cnt is ', cnt, 'content: ', line)
            line_len = len(line.split('\t'))
            # if line_len!=11:
            print('len of line is ', line_len, 'len of user_id_set: ', len(user_id_set), 'current user_id ', user_id)
        # if cnt>=30:
        #     break
print('len of user_id_set is,', len(user_id_set))
print('end prog, cnt is ', cnt, 'time cost ', time.time()-start_t)
