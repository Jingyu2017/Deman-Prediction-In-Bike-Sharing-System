#!/usr/bin/env python2
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None) 
import glob
import seaborn as sns

totol_times=0
df = pd.read_csv('../input/bike_aws/bike_1_7.csv')
df = df['nbBikes'].groupby(df['BikePointId']).count()
nobike = pd.Series(index=df.index)
noDocks = pd.Series(index=df.index)

def addition(infile):
    print infile
    df = pd.read_csv(infile)
    global totol_times
    global nobike
    global noDocks
    totol_times += df.groupby(df['BikePointId']).count().iloc[0,0]
    nBike = df[df['nbBikes']==0]
    nBike = nBike['nbBikes'].groupby(nBike['BikePointId']).count()
    nobike = nobike.add(nBike, fill_value=0)
    nEmptyDocks = df[df['nbEmptyDocks']==0]
    nEmptyDocks = nEmptyDocks['nbEmptyDocks'].groupby(nEmptyDocks['BikePointId']).count()   
    noDocks = noDocks.add(nEmptyDocks, fill_value=0)
    print infile

filenames = glob.glob("../input/bike_aws/*.csv")
for name in filenames:
    addition(name)
    
noDocks=noDocks/5177
nobike=nobike/5177  

nobike.fillna(0,inplace=True)
noDocks.fillna(0,inplace=True)
sns.set(color_codes=True)
import matplotlib.pyplot as plt
sns.set(font_scale=1.5)
sns.distplot(noDocks, kde=False,label="noDocks")  
plt.xlabel('Percentage of Time Stations Are Full') 
plt.ylabel('Number of Stations')

sns.distplot(nobike, kde=False,label="nobikes")    
plt.xlabel('Percentage of Time Stations Are Empty')  
plt.ylabel('Number of Stations')

              
#  noDocks[noDocks>=0.20]    length=10  
#  nobike[nobike>=0.20]     length=173

df = pd.concat([nobike, noDocks], axis=1)
df.columns=['noBike','noDock']
def f(row):
    if row['noBike'] >=0.20 and row['noDock'] >=0.20:
        val = 'black'
    elif row['noBike'] >=0.20:
        val = 'blue'
    elif row['noDock'] >=0.20:
        val = 'red'
    else:
        val = 'green'
    return val

df['color'] = df.apply(f, axis=1)

my_df = pd.read_csv('../output/cluster/station_cluster_output.csv')
my_df = pd.DataFrame(data=my_df, columns=['id','lat','long'])
df2 = pd.merge(my_df,df,left_on='id',right_index=True)

from mapsplotlib import mapsplot as mplt
mplt.register_api_key('AIzaSyBuhp9aMgW--vVDJ5wso-dwVsnZuYJkf6o')

mplt.scatter(df2['lat'], df2['long'], colors=df2['color'])
