# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 17:10:03 2021

@author: syn009
"""

import pandas as pd
import numpy as np
import math
from operator import *

rating = pd.read_csv('C:\\Users\\Administrator\\Desktop\\movie-data\\ml-latest-small\\ratings.csv')
#ratings.csv's columns are user,movieId,rating,timestamp
user = rating['userId']
movieId = rating['movieId']
rating1 = rating['rating'] #人對電影的評分
timestamp = rating['timestamp'] #打分的時間
user_item = dict()#構造一個字典，格式{userid:[movieid,movieid],……}因為一個人會看多部電影
for i in range(len(user)):
    if user[i] not in user_item:
        user_item[user[i]] = set()
        user_item[user[i]].add(movieId[i])
item_user = dict()#建立物品——用戶的倒排表，為了降低表的稀疏性
for i in range(len(user)):
    if movieId[i] not in item_user:
        item_user[movieId[i]] = set()#set結構可以看成一個不重複的陣列
    item_user[movieId[i]].add(user[i])
N = dict()#計數，看看每個電影被看了多少次
C = dict()#得出兩個物品同時被一個用戶看的次數
for i,items in item_user.items():
    for item in items:
        if item not in N:
            N[item] = 0
        N[item]+=1
        for item2 in items:
            if item ==item2:
                continue
            if (item, item2) not in C:
                C[item,item2] = 0
            C[item,item2]+=1
w = dict()#通過C（1,2）/N[1]*N[2] 算出係數表，等於把物品的關聯程度算出來啦;如果想加上時間的因素，就用log加入
for i ,item in C.items():
    w[i[0],i[1]] = C[i[0], i[1]]/math.sqrt(N[i[0]] * N[i[1]] *1.0)
rank = dict()#推薦列表
target_user = 1   #input("write the user you want to recommand:")
train = user_item[target_user]
ralated_item = dict()
for i in w:
    if i[0] in train and i[1] not in train:#找出同時出現的兩部電影，兩部電影有1部被target-user看過
        if i[1] not in ralated_item:
            ralated_item[i[1]] = 0
        ralated_item[i[1]] = w[i[0],i[1]]
print(sorted(ralated_item.items(), key = itemgetter(1), reverse=True)[:10])#列印出前十的。

