# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 12:03:15 2022

@author: syn009
"""

## Import Package
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
## Data 自行定義數據
market_data = {
 'Transaction ID': [1,2,3,4,5,6,7,8],
 'Items':[['T-Shirt','Pants','Jeans','Jersy','Socks','Basketball','Bottle','Shorts'],
 ['T-Shirt','Jeans'],
 ['Jersy','Basketball','Socks','Bottle'],
 ['Jeans','Pants','Bottle'],
 ['Shorts','Basketball'],
 ['Shorts','Jersy'],
 ['T-Shirt'],
 ['Basketball','Jersy'],
 ]}
 
prjRootFolder="C:\\WorkingArea\\Vincent\\DBS\\Y2211-業績分群\\"
readfile="Apriori_result"
df_name = 'CSG'
df_all = pd.read_table(r"C:\\WorkingArea\\Vincent\\DBS\\Y2211-業績分群\\4Apriori_10%.txt",encoding='utf-16')

dataArray = []
df = df_all['CSG']
for i in range(len(df)):
    splitString = df[i].replace(' ','') .split(',')
    dataArray.append(splitString)

ID = [i for i in range(366)]
market_data = {'Transaction ID': ID,
               'Items': dataArray}
## 轉成DataFrame
data = pd.DataFrame(market_data)
## 讓DataFrame 能呈現的寬度大一點
pd.options.display.max_colwidth = 100
## 轉成數值編碼，目前都是字串的組合
data_id = data.drop('Items', 1)
data_items = data.Items.str.join(',')
## 轉成數值
data_items = data_items.str.get_dummies(',')
## 接上Transaction ID
data = data_id.join(data_items)
## 計算支持度 Support
Support_items = apriori(data[['一', '三', '材', '家', '商', '二', '处', '储', '示']], min_support=0.0001, use_colnames = True)
#Support_items = apriori(data[['T-Shirt','Pants','Jeans','Jersy','Socks','Basketball','Bottle','Shorts']], min_support=0.20, use_colnames = True)
## 計算關聯規則 Association Rule
Association_Rules = association_rules(Support_items, metric = 'lift', min_threshold=1)

print(Association_Rules)
Association_Rules.to_csv(prjRootFolder+readfile+"_"+df_name +"_byCSG_all.csv",sep=",",index=False,encoding='utf-16')