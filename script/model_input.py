#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 18:27:34 2018

@author: k1756990
"""

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None) 

# =============================================================================
# creat csv files, check-in and che-out records for each cluster
# ****** takes long time to run this code section*********
# =============================================================================
cluster = pd.read_csv('../output/cluster/station_cluster_output.csv')
clusters = cluster.groupby('cluster')['id'].apply(list)


def write2file(infile,cluster_id):
    df = pd.read_csv(infile)
    df = pd.DataFrame(data=df, columns=["Start Date",'StartStation Id','End Date','EndStation Id'])
    
    df_start = df[np.isin(df['StartStation Id'], clusters[cluster_id])]    
    filename = '../output/cluster/start_from_cluster/start_from_cluster_%s.csv'% cluster_id
    with open(filename, 'a') as f:
        df_start.to_csv(f, header=False)
        
    df_end = df[np.isin(df['EndStation Id'], clusters[cluster_id])]
    filename = '../output/cluster/end_in_cluster/end_in_cluster_%s.csv'% cluster_id
    with open(filename, 'a') as f:
        df_end.to_csv(f, header=False)



import glob
filenames = glob.glob("../input/bike_2_years/*.csv")
for name in filenames:
    for c in range(30):
        write2file(name,c)

# =============================================================================
# 30 clusters check-in check-out for every 15 minutes
# =============================================================================

filename = '../output/cluster/start_from_cluster/start_from_cluster_%s.csv'% 16
df = pd.read_csv(filename,names=
                 ["Start Date",'StartStation Id','End Date','EndStation Id'])
df['Start Date'] = pd.to_datetime(df['Start Date'],format='%d/%m/%Y %H:%M')
df.index = df['Start Date']   
df.sort_index(inplace=True)
df = df.resample('15T', label='right',closed='right').size()


Stations_CheckOut = pd.DataFrame(index=df.index)
Stations_CheckIn = pd.DataFrame(index=df.index)

for i in range(30):
    filename = '../output/cluster/start_from_cluster/start_from_cluster_%s.csv'% i

    df = pd.read_csv(filename,names=
                     ["Start Date",'StartStation Id','End Date','EndStation Id'])
    df['Start Date'] = pd.to_datetime(df['Start Date'],format='%d/%m/%Y %H:%M')
    df.index = df['Start Date']   
    checkOut_bins = df.resample('15T', label='right',closed='right').size()
    Stations_CheckOut['t_out_c_%s'%i] = checkOut_bins


for i in range(30):
    filename = '../output/cluster/end_in_cluster/end_in_cluster_%s.csv'% i

    df = pd.read_csv(filename,names=
                     ["Start Date",'StartStation Id','End Date','EndStation Id'])
    df['End Date'] = pd.to_datetime(df['End Date'],format='%d/%m/%Y %H:%M')
    df.index = df['End Date']   
    checkIn_bins = df.resample('15T', label='right',closed='right').size()
    Stations_CheckIn['t_in_c_%s'%i] = checkIn_bins
  
    
# =============================================================================
# t time slot, t-1 time slot
# =============================================================================
checkIn_minus1 = Stations_CheckIn.shift(1)
checkOut_minus1 = Stations_CheckOut.shift(1)

checkIn_minus1 = checkIn_minus1.rename(columns=lambda x: x.replace('t_', 't-1_'))
checkOut_minus1 = checkOut_minus1.rename(columns=lambda x: x.replace('t_', 't-1_'))

input_attri = pd.concat([Stations_CheckIn, 
                         checkIn_minus1,
                         Stations_CheckOut,
                         checkOut_minus1], axis=1) 

# =============================================================================
# weather
# =============================================================================
weather = pd.read_csv('../input/weather/weather_local_2.csv', index_col=0)    

weather['dt_iso'] = pd.to_datetime(weather['dt_iso'])
weather.drop_duplicates(subset='dt_iso',inplace = True)


weather.index=weather['dt_iso']     
del weather['dt_iso']  
    
input_attri['temp'] =  weather['temp']   
input_attri['humidity'] =  weather['humidity']    
input_attri['wind_speed'] =  weather['wind_speed']   
input_attri['clouds_all'] =  weather['clouds_all']   

weather_minus1 = weather.shift(1)    
input_attri['temp_t-1'] =  weather_minus1['temp']   
input_attri['humidity_t-1'] =  weather_minus1['humidity']    
input_attri['wind_speed_t-1'] =  weather_minus1['wind_speed']   
input_attri['clouds_all_t-1'] =  weather_minus1['clouds_all']     
    
input_attri.fillna(method='ffill', inplace=True)

input_attri = input_attri.iloc[8:] 
input_attri.fillna(0, inplace=True)   

# =============================================================================
# #add more attributes dayofweek(workdays),holiday
# =============================================================================
from sklearn.preprocessing import LabelBinarizer


chri_holiday=['2016-12-22','2016-12-23','2016-12-24','2016-12-25','2016-12-26','2016-12-27','2016-12-28','2016-12-29',
              '2016-12-30','2016-12-31','2017-01-01','2017-01-02','2017-12-21','2017-12-22','2017-12-23','2017-12-24','2017-12-25','2017-12-26','2017-12-27',
              '2017-12-28','2017-12-29','2017-12-31','2017-12-30','2018-01-01','2018-01-02']

input_attri=input_attri [~np.isin(input_attri.index.strftime('%Y-%m-%d'),chri_holiday)]


bank_holiday=['2016-05-30','2016-08-29','2017-04-14',
              '2017-04-14','2017-04-17','2017-05-01','2017-05-29',
              '2017-08-28','2018-03-30','2018-04-02','2018-05-07','2018-05-28']


   
helper = pd.DataFrame(index=input_attri.index)
helper['Date'] = helper.index.strftime('%Y-%m-%d')
    
condi = ~helper.isin(bank_holiday)
helper['Date'] = helper.index.to_series().dt.dayofweek
helper['Date'] = helper['Date'].map(lambda x : 0 if x<=4 else 1)
helper.where(condi,1,inplace=True) 
helper.rename(columns={"Date": "isWeekend"},inplace=True)

input_attri = pd.concat([input_attri,helper], axis=1)    

# =============================================================================
# #add more attributes time
# =============================================================================

#helper = pd.DataFrame(index=input_attri.index)
#helper['time'] = helper.index.time
#
#one_hot = LabelBinarizer().fit_transform(helper['Hour'].values)
#one_hot = pd.DataFrame(data = one_hot,
#             columns=['hour_'+str(i)for i in range(24)],index = helper.index)
#
#input_attri = pd.concat([input_attri,one_hot], axis=1)    

input_attri['time'] = input_attri.index.hour+input_attri.index.minute/60.0



# =============================================================================
# write to csv
# =============================================================================




# =============================================================================
# Y--15min,30min in the future
# =============================================================================


Net = pd.DataFrame(index=input_attri.index)
 
for i in range(30):
    Net['net_15_c_%s'%i] = input_attri['t_in_c_%s'%i]-input_attri['t_out_c_%s'%i]
 
Net = Net.shift(-1)    

a = Net.resample('30T', label='left').sum()   
b = Net.iloc[1:].resample('30T', label='left',base=15).sum()
min30 = pd.concat([a,b], axis=0) 
for i in range(30):
    Net['net_30_c_%s'%i] = min30['net_15_c_%s'%i]
 
# =============================================================================
#  Y--120min in the future
# =============================================================================

c = Net.resample('120T', label='left',base=15).sum() 
d = Net.resample('120T', label='left',base=30).sum()
e = Net.resample('120T', label='left',base=45).sum()
f = Net.resample('120T', label='left',base=60).sum()
g = Net.resample('120T', label='left',base=75).sum()
h = Net.resample('120T', label='left',base=90).sum()
j = Net.resample('120T', label='left',base=105).sum()
k = Net.resample('120T', label='left').sum() 
hour2 = pd.concat([c,d,e,f,g,h,j,k], axis=0) 
for i in range(30):
    Net['net_120_c_%s'%i] = hour2['net_15_c_%s'%i]

Net = Net[:-9]

# =============================================================================
# write input and output to attribute
# =============================================================================
input_attri = input_attri[:-9]


with open('../output/finalInput.csv', 'w') as f:
    input_attri.to_csv(f, index=False)


with open('../output/finalInput_backup_time.csv', 'w') as f:
    input_attri.to_csv(f)

with open('../output/finalInputLabel.csv', 'w') as f:
    Net.to_csv(f, index=False)


with open('../output/finalInputLabel_backup_time.csv', 'w') as f:
    Net.to_csv(f)




