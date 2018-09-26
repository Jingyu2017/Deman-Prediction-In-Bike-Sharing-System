#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 18:21:02 2018

@author: k1756990
"""
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None) 


out7 = pd.read_csv('../output/cluster/7_out.csv',names = ['id', 'Date','number'])
out7 = out7['number'].groupby(out7.id).mean().reset_index().rename(columns={'number':'check_out7'})

#[801 station x mean check out number at morning peak ]
in7 = pd.read_csv('../output/cluster/7_in.csv',names = ['id', 'Date','number'])
in7 = in7['number'].groupby(in7.id).mean().reset_index().rename(columns={'number':'check_in7'})


#[801 station x mean check out number at morning peak ]
out14 = pd.read_csv('../output/cluster/14_out.csv',names = ['id', 'Date','number'])
out14 = out14['number'].groupby(out14.id).mean().reset_index().rename(columns={'number':'check_out14'})


#[801 station x mean check out number at morning peak ]
in14 = pd.read_csv('../output/cluster/14_in.csv',names = ['id', 'Date','number'])
in14 = in14['number'].groupby(in14.id).mean().reset_index().rename(columns={'number':'check_in14'})

df_check = out7.merge(in7,on='id').merge(out14,on='id').merge(in14,on='id')

Location = pd.read_csv('../output/cluster/Location_all.csv')
Location = pd.DataFrame(data=Location,columns=['id','lat','long'])

all_attri = df_check.merge(Location,on='id')


from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

all_attri['a'] = np.log1p(all_attri['check_out7'])
all_attri['b'] = np.log1p(all_attri['check_in7'])
all_attri['c'] = np.log1p(all_attri['check_out14'])
all_attri['d'] = np.log1p(all_attri['check_in14'])
#X = MinMaxScaler().fit_transform(all_attri[['check_out7','check_in7','check_out14','check_in14','lat','long']])
X = MinMaxScaler().fit_transform(all_attri[['a','b','c','d','lat','long']])
X[:,4] = X[:,4] * 2
X[:,5] = X[:,5] * 2



import sklearn.metrics as metrics
import matplotlib.pyplot as plt
def kmeans(X, n_cluster):

    res = KMeans(n_clusters=n_cluster,init='k-means++',
                            random_state=50).fit(X)

    silhouette_score = metrics.silhouette_score(
        X, res.labels_, metric='euclidean')
    calinski_harabaz_score = metrics.calinski_harabaz_score(
        X, res.labels_)

    
    return {'cluster': res.labels_, 
            'silhouette_score': silhouette_score, 
            'calinski_harabaz_score': calinski_harabaz_score}
    
    

silhouette_score = []
calinski_harabaz_score = []

for i in range(20, 51):

    clusters_result = kmeans(X, i)
    silhouette_score.append(clusters_result['silhouette_score'])
    calinski_harabaz_score.append(clusters_result['calinski_harabaz_score'])



fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(range(20, 51), silhouette_score,lw=5,color='g')
ax1.set_xlabel('Number of clusters', fontsize=20)
ax1.set_ylabel('Silhouette Score',  fontsize=20,color='g') 


ax2 = ax1.twinx()
ax2.plot(range(20, 51), calinski_harabaz_score, 'orange',lw=5)
ax2.set_ylabel('Calinski Harabaz Score', fontsize=20,color='orange')  

plt.show()