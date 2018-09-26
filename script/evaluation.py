#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np
import seaborn as sns
import pandas as pd
pd.set_option('display.max_columns', None)
from sklearn.metrics import r2_score 
import matplotlib.pyplot as plt
import seaborn as sns
true15 = pd.read_csv('../output/Prediction/true15.csv',index_col=0)
mdt15 = pd.read_csv('../output/Prediction/mdt15.csv',index_col=0)
mrf15 = pd.read_csv('../output/Prediction/mrf15.csv',index_col=0)
srf15 = pd.read_csv('../output/Prediction/srf15.csv',index_col=0)
nn15 = pd.read_csv('../output/Prediction/nn15.csv',index_col=0)


true30 = pd.read_csv('../output/Prediction/true30.csv',index_col=0)
mdt30 = pd.read_csv('../output/Prediction/mdt30.csv',index_col=0)
mrf30 = pd.read_csv('../output/Prediction/mrf30.csv',index_col=0)
srf30 = pd.read_csv('../output/Prediction/srf30.csv',index_col=0)
nn30 = pd.read_csv('../output/Prediction/nn30.csv',index_col=0)

true120 = pd.read_csv('../output/Prediction/true120.csv',index_col=0)
mdt120 = pd.read_csv('../output/Prediction/mdt120.csv',index_col=0)
mrf120 = pd.read_csv('../output/Prediction/mrf120.csv',index_col=0)
srf120 = pd.read_csv('../output/Prediction/srf120.csv',index_col=0)
nn120 = pd.read_csv('../output/Prediction/nn120.csv',index_col=0)

# =============================================================================
# barplot for 3 models for 15,30,120 minutes
# =============================================================================

data = {'prediction window':['15 mins','15 mins','15 mins','15 mins','30 mins','30 mins',
                             '30 mins','30 mins','120 mins','120 mins','120 mins','120 mins'],
        'score':[r2_score(true15, srf15),r2_score(true15, mdt15),r2_score(true15, mrf15),r2_score(true15, nn15),
           r2_score(true30, srf30),r2_score(true30, mdt30),r2_score(true30, mrf30),r2_score(true30, nn30),
           r2_score(true120, srf120),r2_score(true120, mdt120),r2_score(true120, mrf120),r2_score(true120, nn120)],
           'model':('Univariate Random Forests', 'Multivariate Decision Trees','Multivariate Random Forests','Neural Network')*3
}
df = pd.DataFrame(data=data)

sns.set_style("white")

plt.figure(figsize=(8,8))

sns.set(font_scale=1.7)
plt.ylim(0, 1.0)

ax= sns.barplot(x='prediction window', y='score', hue='model', data=df)
plt.legend(bbox_to_anchor=(0.6, 1.24), loc=1, borderaxespad=0.)
plt.ylabel(r'$R^2$ score')
# =============================================================================
# barplot for each cluster
# =============================================================================

score15 = pd.DataFrame()

score15['URF']=r2_score(true15, srf15,multioutput='raw_values')
score15['MDT']=r2_score(true15, mdt15,multioutput='raw_values')
score15['MRF']=r2_score(true15, mrf15,multioutput='raw_values')
score15['NN']=r2_score(true15, nn15,multioutput='raw_values')


score30 = pd.DataFrame()

score30['URF']=r2_score(true30, srf30,multioutput='raw_values')
score30['MDT']=r2_score(true30, mdt30,multioutput='raw_values')
score30['MRF']=r2_score(true30, mrf30,multioutput='raw_values')
score30['NN']=r2_score(true30, nn30,multioutput='raw_values')


score120 = pd.DataFrame()

score120['URF']=r2_score(true120, srf120,multioutput='raw_values')
score120['MDT']=r2_score(true120, mdt120,multioutput='raw_values')
score120['MRF']=r2_score(true120, mrf120,multioutput='raw_values')
score120['NN']=r2_score(true120, nn120,multioutput='raw_values')


for i in range(30):
    a = pd.DataFrame(score15.T[i])
    a['prediction window']='15 mins'
    
    b = pd.DataFrame(score30.T[i])
    b['prediction window']='30 mins'
    
    c = pd.DataFrame(score120.T[i])
    c['prediction window']='120 mins'
    
    d = pd.concat([a,b,c],axis=0)
    
    d.reset_index(inplace=True)
    d.columns=['model','R2 score','Prediction window']
    
    
    plt.figure(figsize=(7,6))
    sns.set_style("whitegrid")
    sns.set(font_scale=1.8)
    plt.ylim(0, 1.0)
    
    plt.title('Cluster %s'%i,fontdict ={ 'fontweight':'bold'})
    ax= sns.barplot(x='Prediction window', y='R2 score', hue='model', data=d)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.ylabel(r'$R^2$ score')
    plt.show()








