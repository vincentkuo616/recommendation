# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 14:30:44 2021

@author: vincentkuo
"""



import pandas as pd

df = pd.read_csv('C:\\Users\\vincentkuo\\Downloads\\ml-25m\\ml-25m\\ratings_100.csv')

print(df['rating'].mean())

# 只取平均值以上的數據，作為喜歡的列表
df = df[df['rating'] > df['rating'].mean()].copy()
print(df.head())

df_group = df.groupby(['userId'])['movieId'].apply(lambda x: ' '.join([str(m) for m in x])).reset_index()
print(df_group.head())

#df_group.to_csv("C:\\Users\\vincentkuo\\Downloads\\ml-25m\\ml-25m\\movielens_uid_movieids.csv", index=False)

import findspark
findspark.init()

from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName('PySpark Item2vec') \
    .getOrCreate()

sc = spark.sparkContext

df = pd.read_csv("C:\\Users\\vincentkuo\\Downloads\\ml-25m\\ml-25m\\movielens_uid_movieids.csv", header=True)
print(df.show(5))
# df = pd.read_csv("C:\\Users\\vincentkuo\\Downloads\\ml-25m\\ml-25m\\movielens_uid_movieids.csv")

from pyspark.sql import functions as F
from pyspark.sql import types as T

# 把非常的字符串格式變成List的形式
df = df.withColumn('movie_ids', F.split(df.movieId, " "))

# httpsL//spark.apache.org/docs/2.4.6/ml-features.html#word2vec

from pyspark.ml.feature import Word2Vec

word2Vec = Word2Vec(vectorSize = 5, minCount = 0, inputCol = 'movie_ids', outputCol = 'movie_2vec')

model = word2Vec.fit(df)
# 不計算每個USER的Embedding, 而是計算item的Embedding
model.getVectors().show(3, truncate=False)
model.getVectors().select('word', 'vector').toPandas().to_csv("C:\\Users\\vincentkuo\\Downloads\\ml-25m\\ml-25m\\movielens_movie_embedding.csv", index=False)

df_embedding = pd.read_csv("C:\\Users\\vincentkuo\\Downloads\\ml-25m\\ml-25m\\movielens_movie_embedding.csv")
print(df_embedding.head(3))

df_movie = pd.read_csv("C:\\Users\\vincentkuo\\Downloads\\ml-25m\\ml-25m\\movies.csv")

df_merge = pd.merge(left=df_embedding, right=df_movie,left_on='word',right_on='movieId')

import numpy as np
import json

df_merge['vector'] = df_merge['vector'].map(lambda x : np.array(json.load(x)))
#隨機挑選一個電影: 4018 What Women Want (2000)
movie_id = 4018
df_merge.loc[df_merge['movieId']==movie_id]
movie_embedding = df_merge.loc[df_merge['movieId']==movie_id, 'vector'].iloc[0]
print(movie_embedding)

# 餘弦相似度
from scipy.spatial import distance
df_merge['sim_value'] = df_merge['vector'].map(lambda x : 1 - distance.cosine(movie_embedding, x))

# 按照餘弦相似度降序排列, 查詢前10條
print(df_merge.sort_values(by='sim_value', ascending=False)[['movie_id', 'title', 'genres', 'sim_value']].head(10))


