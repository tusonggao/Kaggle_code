import time
import os
import numpy as np
import pandas as pd

exposure_file = './data/algo.qq.com_641013010_testa/testA/imps_log/totalExposureLog.out'
# 10 列 102386695 行
# 广告请求ID有82085734个不同的  广告位id有329个不同的   用户ID有1341958不同的
# 曝光广告有509280个不同的   曝光广告素材尺寸64个 曝光广告出价bid有19036个不同值
# 曝光广告pctr有794225个不同的值  曝光广告quality_ecpm有1008232个不同值  曝光广告totalEcpm有6585693个不同值
# 广告请求ID  广告请求时间  广告位id  用户ID  曝光广告  曝光广告素材尺寸  曝光广告出价bid  曝光广告pctr  曝光广告quality_ecpm  曝光广告totalEcpm
# line_cnt is  100000 content:  76535481	1550408116	39	926292	118401	36	45	18.812	376.24	1222.24
# line_cnt is  200000 content:  45340639	1550368494	25	660821	206786	30	11	13.492	269.84	418.252
# line_cnt is  300000 content:  29906070	1550333927	209	447125	476095	64	27	8.431	168.62	396.257

# final date_count is  {'2019-02-16': 9178, '2019-02-17': 3641843, '2019-02-18': 3662297, '2019-02-19': 3614462,
# '2019-02-20': 3375771, '2019-02-21': 3180386, '2019-02-22': 3200624, '2019-02-23': 3535602, '2019-02-24': 3403865,
# '2019-02-25': 2875131, '2019-02-26': 2909223, '2019-02-27': 2962564, '2019-02-28': 2963833, '2019-03-01': 3180607,
# '2019-03-02': 3687014, '2019-03-03': 3531212, '2019-03-04': 2969010, '2019-03-05': 3003831, '2019-03-06': 2999041,
# '2019-03-07': 3019025,
# '2019-03-08': 3191187, '2019-03-09': 3809474, '2019-03-10': 3680312, '2019-03-11': 3046077, '2019-03-12': 3128640,
# '2019-03-13': 3110763, '2019-03-14': 3188419, '2019-03-15': 3346954, '2019-03-16': 3949803, '2019-03-17': 3849068,
# '2019-03-18': 3221264, '2019-03-19': 3140215}


user_file = './data/algo.qq.com_641013010_testa/testA/user/user_data'
# 11 列 1396718 行  每一行的user_id都是不同的
# 用户ID有1396718个不同值   年龄92个不同的值  性别有3个不同值  地域面上有708457个 实际13646个不同值
# 婚恋状态有19个不同值 多值逗号分隔   学历有8个不同值  消费能力有3个不同值
# 设备有4个不同值   工作状态有7个不同值   连接类型有5个不同值  行为兴趣有32631个不同值
# 用户ID  年龄  性别  地域  婚恋状态  学历  消费能力  设备  工作状态  连接类型  行为兴趣

# line_cnt is  1000000 content:  884336	819	3	-1,-1,1231,12048,11524,-1	0	6	1	3	0	2	0


ad_static_feature_file = './data/algo.qq.com_641013010_testa/testA/ad_static_feature.out'
# 7 列 735911 行
# 广告id有735911个不同值   广告账户id有29737个不同值  商品id有32921个不同值（有可能有多值）
# 商品类型有17个不同值（没有多值）  广告行业id有247个不同值（有多值）  素材尺寸有66个不同值（可能有多值）
# 广告ID  创建时间    广告账户id   商品id   商品类型   广告行业id   素材尺寸
# line_cnt is 700000 line is 435633	1552031632	23185	30103	18	117	64


ad_operation_file = './data/algo.qq.com_641013010_testa/testA/ad_operation.dat'
# 7 列 760866 行
# 广告ID有38843个不同值  操作类型有2个不同值：1修改 2新建
# 修改字段有4个不同值
# 广告ID   创建时间    操作类型    修改字段    操作后字段值    人群定向


start_t = time.time()
print('start prog')
date_count = {}
with open(exposure_file) as file:
    cnt = 0
    user_id_set = set()
    for line in file:
        cnt += 1
        # if cnt >= 500:
        #     break
        time_str = line.split('\t')[1]
        real_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time_str)))
        day_str = real_time_str[:10]
        date_count[day_str] = date_count.get(day_str, 0) + 1
        if cnt%5000==0:
            print('cnt is', cnt)
            print('real_time_str is', real_time_str, 'real_time_str[:10] is', day_str)
            print('current date_count is ', date_count)

print('final date_count is ', date_count)

######################################################################################

# start_t = time.time()
# print('start prog')
# with open(exposure_file) as file:
#     cnt = 0
#     user_id_set = set()
#     for line in file:
#         cnt += 1
#         if cnt >= 5:
#             break
#         user_id = line.split('\t')[1]
#         place_ids = set(user_id.split(','))
#         if len(place_ids)>=2:
#             print('find a comma in this field， user_id is ', user_id)
#             break
#         # user_id_set.add(user_id)
#         user_id_set |= place_ids
#         if cnt%100000==0:
#             print('line_cnt is ', cnt, 'line is', line)
#             # print('user id is ', user_id, 'len of place_ids is ', len(place_ids))
#             line_len = len(line.split('\t'))
#             # if line_len!=11:
#             print('len of line is ', line_len, 'len of user_id_set: ', len(user_id_set), 'current user_id ', user_id,
#                   'line_cnt is', cnt)
#             # print('user id is ', user_id, 'len of place_ids is ', len(place_ids),
#             #       'len of user_id_set ', len(user_id_set), 'line num:', cnt)
#         # if cnt>=30:
#         #     break
# print('len of user_id_set is,', len(user_id_set))
# # print('content of user_id_set is ', sorted(list(user_id_set)))
# print('end prog, cnt is ', cnt, 'time cost ', time.time()-start_t)