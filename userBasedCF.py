# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 16:29:22 2021

@author: syn009
"""


import numpy as np
import math
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import paired_distances,cosine_similarity
import time


''' 測試練習區
'''
columnsName = ['user_0','user_1','user_2','user_3','user_4','user_5','user_6','user_7','user_8','user_9']
indexName = ['movie_0','movie_1','movie_2','movie_3','movie_4','movie_5','movie_6','movie_7','movie_8','movie_9']

data=[[0, 0, 3, 4, 2, 1, 2, 0, 5, 1],
      [3, 0, 1, 3, 0, 0, 0, 0, 0, 0],
      [0, 3, 0, 4, 0, 2, 0, 0, 0, 2],
      [5, 2, 3, 2, 0, 4, 3, 3, 0, 0],
      [0, 5, 5, 0, 0, 0, 0, 0, 5, 4],
      [0, 0, 0, 0, 4, 0, 4, 2, 3, 0],
      [4, 4, 0, 0, 4, 4, 3, 4, 0, 4],
      [5, 0, 4, 2, 3, 0, 3, 3, 3, 3],
      [0, 3, 0, 0, 5, 5, 0, 4, 0, 0],
      [2, 0, 0, 0, 0, 0, 0, 0, 4, 0]]

df = pd.DataFrame(data=data,columns=columnsName,index=indexName)


#df = pd.read_excel("D://Temp//Vincent//臨時檔案//cust_item_filter_noweight.xlsx",index_col='CUSTOMERCODE')
#df = df.fillna(0).T

df_dict = {}
for i in df.columns:
    buyList=[]
    for j in df.index:
        if(df[i][j]>0):
            buyList.append(j)
    df_dict[i]=buyList


def UserSimilarity(train , IIF = False, m=1):
    # IIF 是否對 過於熱門即 購買人數過於多的物品 在計算用戶相似度的時候進行懲罰
    # 因為很多用戶對之間並沒有對相同的物品產生過行為，只計算對相同物品產生過行為的用戶之間的相似度。
    # 采用余弦相似度
    # 建立倒排表，對每個物品保存只對其產生過行為的用戶列表。
    item_users = dict() # 物品-用戶 倒排表
    for u, items in train.items():
        for i in items:
            # 這裏將 item_users.keys() 改為 item_users , 文中例子 應該用set 或 list存，而不是dict:
            if i not in item_users:
                item_users[i] = set()
            item_users[i].add(u)

    if(m==1):
        # 法一 使用次數 不考慮量 購買了A 3次 跟 5次 都叫做1
        # 建立倒排矩陣
        C = dict() # key 用戶對 value 購買同一物品的次數
        N = dict() # N(u) 表示用戶購買的 商品數 {'A': 3, 'B': 2, 'C': 2, 'D': 3}
        for i,users in item_users.items():
            for u in users:
                if u not in N.keys():
                    N[u] = 0
                N[u] += 1*1
                for v in users:
                    if u == v:
                        continue
                    if (u,v) not in C.keys():
                        C[u,v] = 0
                    if IIF:
                        # len(users) 表示購買此物品的用戶數，越熱門，購買用戶越多,C[u,v] 就越小
                        # 相當於之前的分子是相交個數，現在是
                        C[u,v] += 1 / math.log(1 + len(users))
                    else:
                        C[u,v] += 1
        W = dict()
        for co_user, cuv in C.items():
            W[co_user] = cuv / math.sqrt(N[co_user[0]]*N[co_user[1]])
    else:
        # 法二 考慮量 購買了A 3次 跟 5次 一個叫做3 一個叫5
        # 建立倒排矩陣
        C = dict() # key 用戶對 value 購買同一物品的次數
        N = dict() # N(u) 表示用戶購買的 商品數 {'A': 3, 'B': 2, 'C': 2, 'D': 3}
        for i,users in item_users.items():
            for u in users:
                if u not in N.keys():
                    N[u] = 0
                N[u] += df[u][i]*df[u][i]
                for v in users:
                    if u == v:
                        continue
                    if (u,v) not in C.keys():
                        C[u,v] = 0
                    if IIF:
                        # len(users) 表示購買此物品的用戶數，越熱門，購買用戶越多,C[u,v] 就越小
                        # 相當於之前的分子是相交個數，現在是
                        C[u,v] += df[u][i]*df[v][i] / math.log(1 + len(users))
                    else:
                        C[u,v] += df[u][i]*df[v][i]
        W = dict()
        print(C)
        for co_user, cuv in C.items():
            W[co_user] = cuv / math.sqrt(N[co_user[0]]*N[co_user[1]])
    
    return W
    
def UserCFRecommend(user,train,W,k):
    # rvi 代表用戶v對物品i的權重
    rvi = 1 
    rank = dict()
    interacted_items = train[user]
    related_user=[]
    # 和 A 有相似度的用戶 ，B,C,D
    for co_user,sim in W.items():
        if co_user[0] == user:
            related_user.append((co_user[1],sim))
    # v : 有相似度的用戶 , wuv : 用戶間相似度 
    for v , wuv in sorted(related_user , key = lambda a:a[1], reverse = True)[0:k]:
        for item in train[v]:
            if item in interacted_items:
                continue
            else:
                # 還是得初始化，才可以賦值
                if item not in rank.keys():
                    rank[item] = 0 
                rank[item] += wuv*rvi
    return rank

if __name__=='__main__':
    # train = {'A':('a','b','d'),'B':('a','c'),'C':('b','e'),'D':('c','d','e')}
    # W = UserSimilarity (df_dict , IIF = False , m=2)
    # rank = UserCFRecommend('user_1', df_dict, W , k = 3)
    # print (rank)
    print("\n @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Main Start @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ")
    MainStartTime = time.time()
    
    
    W = UserSimilarity (df_dict , IIF = False , m=2)
    rank = UserCFRecommend(16, df_dict, W , k = 3)
    print (rank)
#    pd.DataFrame(list(W.items()),columns=['AB','SIM']).to_csv("D://Temp//Vincent//臨時檔案//cust_item_result2_noweight2.csv",index=False)
    
    
    MainEndTime = time.time()
    print(" Total Cost: "+"{}".format(round(MainEndTime-MainStartTime,2))+"s" )
    print("\n @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Main END @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ")
    
    
#    df = pd.read_excel("D://Temp//Vincent//臨時檔案//cust_item_filter.xlsx",index_col='CUSTOMERCODE')
#    df = df.fillna(0).T
#    
#    df_dict = {}
#    for i in df.columns:
#        buyList=[]
#        for j in df.index:
#            if(df[i][j]>0):
#                buyList.append(j)
#        df_dict[i]=buyList
#    
#    print("\n @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Main Start @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ")
#    MainStartTime = time.time()
#    
#    
#    W = UserSimilarity (df_dict , IIF = False , m=2)
#    pd.DataFrame(list(W.items()),columns=['AB','SIM']).to_csv("D://Temp//Vincent//臨時檔案//cust_item_result2.csv",index=False)
#    
#    
#    MainEndTime = time.time()
#    print(" Total Cost: "+"{}".format(round(MainEndTime-MainStartTime,2))+"s" )
#    print("\n @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Main END @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ")

# print(recommend_movies('user_0', 3, 4))