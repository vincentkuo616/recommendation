# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 11:19:28 2022

@author: syn009
"""

## Import package
from apyori import apriori
import pandas as pd
## Data 自行定義數據
market_data = [['T-Shirt','Pants','Jeans','Jersy','Socks','Basketball','Bottle','Shorts'],
 ['T-Shirt','Jeans'],
 ['Jersy','Basketball','Socks','Bottle'],
 ['Jeans','Pants','Bottle'],
 ['Shorts','Basketball'],
 ['Shorts','Jersy'],
 ['T-Shirt'],
 ['Basketball','Jersy'],
 ]

prjRootFolder="C:\\WorkingArea\\Vincent\\DBS\\Y2211-業績分群\\"
readfile="Apriori_result"
df_all = pd.read_table(r"C:\\WorkingArea\\Vincent\\DBS\\Y2211-業績分群\\4Apriori_10%.txt",encoding='utf-16')

dataArray = []
df = df_all['CSG']
for i in range(len(df)):
    splitString = df[i].replace(' ','') .split(',')
    dataArray.append(splitString)

#print(dataArray)

association_rules = apriori(dataArray, min_support=0.001, min_confidence=0.001, min_lift=2, max_length=2)
association_rules_metric = apriori(dataArray, min_support=0.001, min_confidence=0.001, min_lift=2, max_length=2)
association_results = list(association_rules)
##print(association_results )
for product in association_results:
 #print(product) # ex. RelationRecord(items=frozenset({'Basketball', 'Socks'}), support=0.25, ordered_statistics=[OrderedStatistic(items_base=frozenset({'Basketball'}), items_add=frozenset({'Socks'}), confidence=0.5, lift=2.0), OrderedStatistic(items_base=frozenset({'Socks'}), items_add=frozenset({'Basketball'}), confidence=1.0, lift=2.0)])
 pair = product[0] 
 ##print(pair) ## ex. frozenset({'Basketball', 'Socks'})
 products = [x for x in pair]
 print(products) # ex. ['Basketball', 'Socks']
 print("Rule: " + products[0] + " →" + products[1])
 print("Support: " + str(product[1]))
 print("Lift: " + str(product[2][0][3]))
 print("==================================")