#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
infile = "../input/bike_2_years/07JourneyDataExtract25May2016-31May2016.csv"
"""

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None) 
from matplotlib import pyplot as plt
import seaborn as sns

# =============================================================================
# part 1 how many docking stations for each week
# =============================================================================
nums=[]
names=[]
def howmany(infile):
    df_r = pd.read_csv(infile)
    df_r = pd.DataFrame(data=df_r, columns=["EndStation Id",'StartStation Id'])
    
    s = set( list(df_r["EndStation Id"].unique())+ list(df_r["StartStation Id"].unique()))
    num = len(s)
    nums.append(num)
    names.append(infile)
import glob
filenames = glob.glob("../input/bike_2_years/*.csv")
for name in filenames:
    howmany(name)
    
    



d = {'col1': names, 'col2': nums}
df_p = pd.DataFrame(data=d)
df_p['order'] = df_p['col1'].map(lambda x: int(x.split("Jour")[0].split('/')[-1]))

df_p.sort_values(by=['order'],inplace=True)
df_p.reset_index(inplace=True)
 
sns.set_style("darkgrid")
plt.plot(df_p['col2'])
plt.xlabel('Week',fontsize=18)
plt.ylabel('Number of Docking Stations',fontsize=18) 
plt.xticks(fontsize=17)
plt.yticks(np.arange(765, 791, step=4),fontsize=17)


# =============================================================================
# part 2 find out all the docking stations ever used 
# =============================================================================


def stationAll(infile):

    df_s = pd.read_csv(infile)
    df_s = pd.DataFrame(data=df_s, columns=["EndStation Id",'StartStation Id','EndStation Name','StartStation Name'])
       
    t = df_s.groupby(["EndStation Id",'EndStation Name']).size().reset_index().rename(columns={0:'count'})
    with open('../output/cluster/stations_id_name.csv', 'a') as f:
        t.to_csv(f, header=False)  
import glob
filenames = glob.glob("../input/bike_2_years/*.csv")
for name in filenames:
    stationAll(name)

sta = pd.read_csv('../output/cluster/stations_id_name.csv',names=['d','id','name','size'])
pop = sta['id'].unique()
 
df2018 = pd.read_csv('../input/DockingLocation.csv')
list2018 = df2018['id'].unique()

temp=list(set(pop)-set(list2018))


df_h = pd.DataFrame(data={'col1': temp})
pop_up = pd.merge(df_h,sta,left_on='col1',right_on='id',how='left').drop_duplicates(subset='id')
with open('../output/cluster/pop_ups_id.csv', 'w') as f:
    pop_up.to_csv(f, columns=['id','name'])  

# =============================================================================
# part 3 convert british national system to wgs84,and merge into one complete reference
# =============================================================================
import pyproj as proj

# setup your projections
crs_wgs = proj.Proj(init='epsg:4326') # assuming you're using WGS84 geographic
crs_bng = proj.Proj(init='epsg:27700') # use a locally appropriate projected CRS

bng = pd.read_csv('../input/pop_ups_id.csv')
bng['x'] = bng['x'].map(lambda x: np.float64(x.replace(',','')))
bng['y'] = bng['y'].map(lambda x: np.float64(x.replace(',','')))

bng['long'], bng['lat'] = proj.transform(crs_bng, crs_wgs, list(bng.x), list(bng.y))

with open('../output/cluster/Location_all.csv', 'w') as f:
    df2018.to_csv(f, columns=['id','lat','long']) 
with open('../output/cluster/Location_all.csv', 'a') as f:
    bng.to_csv(f, columns=['id','lat','long'],header=False)    

# =============================================================================
# part 4 check-out number during morning peak on Thursdays 
# =============================================================================
def write2file(infile):
    df = pd.read_csv(infile)
    df = pd.DataFrame(data=df, columns=["Start Date","StartStation Id","End Date","EndStation Id"])
    df['Start Date'] = pd.to_datetime(df['Start Date'],format='%d/%m/%Y %H:%M')  
    df['End Date'] = pd.to_datetime(df['End Date'],format='%d/%m/%Y %H:%M') 

 #For each station, each Thursday
    #check-out number during morning peak on Thursdays (avoid holidays which often take place on Mondays and Fridays)
    df_a = df[(df['Start Date'].dt.hour>=7) & (df['Start Date'].dt.hour<=9) & (df['Start Date'].dt.dayofweek==3)]
    df_a = df_a.groupby(['StartStation Id',df['Start Date'].dt.date]).size().reset_index()
    with open('../output/cluster/7_out.csv', 'a') as f:
            df_a.to_csv(f, header=False)
            
    # check-in   during morning peak on Thursdays 
    df_b = df[(df['End Date'].dt.hour>=7)&(df['End Date'].dt.hour<=9) & (df['End Date'].dt.dayofweek==3)]
    df_b = df_b.groupby(['EndStation Id',df['End Date'].dt.date]).size().reset_index()
    with open('../output/cluster/7_in.csv', 'a') as f:
            df_b.to_csv(f, header=False)  
    
        
    #check-out number during  peak on Sat & Sunday 
    df_c = df[(df['Start Date'].dt.hour>=13)&(df['Start Date'].dt.hour<=15) & (df['Start Date'].dt.dayofweek>=5)]
    df_c = df_c.groupby(['StartStation Id',df['Start Date'].dt.date]).size().reset_index()
    with open('../output/cluster/14_out.csv', 'a') as f:
            df_c.to_csv(f, header=False)
    
    #check-in number during peak on Sat & Sunday 
    df_d = df[(df['End Date'].dt.hour>=13)&(df['End Date'].dt.hour<=15) & (df['End Date'].dt.dayofweek>=5)]
    df_d = df_d.groupby(['EndStation Id',df['End Date'].dt.date]).size().reset_index()
    with open('../output/cluster/14_in.csv', 'a') as f:
            df_d.to_csv(f, header=False)        
            

import glob
filenames = glob.glob("../input/bike_2_years/*.csv")
for name in filenames:
    write2file(name)
    


# =============================================================================
# part 5 attributes for clustering (check-in/out: 4 attributes, lat/long)
# =============================================================================
    
#[801 station x mean check out number at morning peak ]     
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


# =============================================================================
# part 6 KMeans clustering
# =============================================================================



from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

all_attri['a'] = np.log1p(all_attri['check_out7'])
all_attri['b'] = np.log1p(all_attri['check_in7'])
all_attri['c'] = np.log1p(all_attri['check_out14'])
all_attri['d'] = np.log1p(all_attri['check_in14'])
X = MinMaxScaler().fit_transform(all_attri[['a','b','c','d','lat','long']])
X[:,4] = X[:,4] * 2
X[:,5] = X[:,5] * 2
kmeans = KMeans(n_clusters=30, init='k-means++', random_state=50)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

all_attri['cluster'] = y_kmeans


plt.figure(figsize=(16,12))
plt.scatter(all_attri['long'], all_attri['lat'], c=all_attri['cluster'], s=50, cmap='tab20')
plt.show()


with open('../output/cluster/station_cluster_output.csv', 'w') as f:
    all_attri.to_csv(f)



# =============================================================================
# part 7 visual
# =============================================================================
my_df = pd.read_csv('../output/cluster/station_cluster_output.csv')
from mapsplotlib import mapsplot as mplt

mplt.register_api_key('AIzaSyBuhp9aMgW--vVDJ5wso-dwVsnZuYJkf6o')
mplt.polygons(my_df['lat'],my_df['long'],my_df['cluster'])



# =============================================================================
# part 8 isual (each cluster)
# =============================================================================
my_df = pd.read_csv('../output/cluster/station_cluster_output.csv')

for i in range(30): 
    
    fig = plt.figure(figsize=(9,7))
    alphas = my_df['cluster'].map(lambda x: 1 if x==i else 0.15)
    rgba_colors = np.zeros((800,4))
    # for red the first column needs to be one
    rgba_colors[:,0] = 1.0
    # the fourth column is  alphas
    rgba_colors[:, 3] = alphas
#    plt.gca().set_xticks([])
#    plt.gca().set_yticks([])
    ax = fig.add_subplot(111) 
    ax.scatter(my_df['long'], my_df['lat'], color=rgba_colors, s=20)
    ax.axis('off')
    plt.savefig('../output/cluster_%s'%i)
 
 
# =============================================================================
# visual (3;15 cluster)
# ============================================================================= 
my_df = pd.read_csv('../output/cluster/station_cluster_output.csv')
fig = plt.figure(figsize=(9,7))
alphas = my_df['cluster'].map(lambda x: 1 if (x==23) else 0.15)
rgba_colors = np.zeros((800,4))
# for red the first column needs to be one
rgba_colors[:,0] = 1.0
# the fourth column is  alphas
rgba_colors[:, 3] = alphas
#    plt.gca().set_xticks([])
#    plt.gca().set_yticks([])
ax = fig.add_subplot(111) 
ax.scatter(my_df['long'], my_df['lat'], color=rgba_colors, s=20)
ax.axis('off')
plt.savefig('../output/cluster_%s'%i)



l= [3,15,11,14,23]
all_attri['cluster'] = all_attri['cluster'].map(lambda x: 0 if x not in l else x)


fig = plt.figure(figsize=(11,9))
ax = fig.add_subplot(111) 
ax.scatter(all_attri['long'], all_attri['lat'], c=all_attri['cluster'], s=45,cmap='Set2')
ax.axis('off')
ax.text(-0.092,51.522, 'C3', fontsize=21,horizontalalignment='center', verticalalignment='center')
ax.text(-0.11,51.514, 'C15', fontsize=21,horizontalalignment='center', verticalalignment='center')

ax.text(-0.09,51.5, 'C11', fontsize=21,horizontalalignment='center', verticalalignment='center')
ax.text(-0.08,51.535, 'C14', fontsize=21,horizontalalignment='center', verticalalignment='center')
ax.text(-0.08,51.508, 'C23', fontsize=21,horizontalalignment='center', verticalalignment='center')























