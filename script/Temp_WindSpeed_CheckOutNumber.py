# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 00:11:12 2018

@author: li
"""
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None) 

weather = pd.read_csv('../input/weather/weather_local_2.csv')


weather['dt_iso'] = pd.to_datetime(weather['dt_iso'])
weather.index = weather['dt_iso']
del weather['dt_iso']

weather.index=weather.index.date
hum = weather['humidity'].groupby(weather.index).median().reset_index()
temp = weather['temp'].groupby(weather.index).median().reset_index()
speed = weather['wind_speed'].groupby(weather.index).median().reset_index()
clouds = weather['clouds_all'].groupby(weather.index).median().reset_index()

bank_holiday=['2016-05-30','2016-08-29','2016-12-21','2016-12-22','2016-12-23',
              '2016-12-26','2016-12-27','2016-12-28','2016-12-29',
              '2016-12-30','2017-01-02','2017-04-14',
              '2017-04-14','2017-04-17','2017-05-01','2017-05-29',
              '2017-08-28','2017-12-21','2017-12-22',
              '2017-12-25','2017-12-26','2017-12-27','2017-12-28','2017-12-29','2018-01-01','2018-01-02',
              '2018-03-30','2018-04-02','2018-05-07','2018-05-28']


def write2file(infile):
    

    df = pd.read_csv(infile)
    df = pd.DataFrame(data=df, columns=["Start Date"])
    df['Start Date'] = pd.to_datetime(df['Start Date'],format='%d/%m/%Y %H:%M')    
    df['date'] = df['Start Date'].dt.date
    df['day_of_week']=df['Start Date'].dt.dayofweek
    
    df_weekend = df[(df['day_of_week']>4)]
    df_weekend_size = df_weekend.groupby(df_weekend['date']).size().reset_index()
    df_weekend_size.columns=['date','size']
    
    with open('../output/number_Trips_weekend.csv', 'a') as f:
        df_weekend_size.to_csv(f, header=False)   
    
    
    df = df[(df['day_of_week']<=4)]
    
    df_size = df.groupby(df['date']).size().reset_index()
    df_size.columns=['date','size']
    
    with open('../output/number_Trips.csv', 'a') as f:
        df_size.to_csv(f, header=False)
    

import glob
filenames = glob.glob("../input/bike_2_years/*.csv")
for name in filenames:
    write2file(name)
    

df_weekday = pd.read_csv('../output/number_Trips.csv',names = ['index', 'size'])

workday = df_weekday[~np.isin(df_weekday['index'], bank_holiday)]
holiday = df_weekday[np.isin(df_weekday['index'], bank_holiday)]
workday['type'] = 'Weekday'
holiday['type'] = 'Holiday'

weekend = pd.read_csv('../output/number_Trips_weekend.csv',names = ['index', 'size'])
weekend['type'] = 'Holiday'

df = pd.concat([workday,holiday,weekend], ignore_index=True)

with open('../output/hum.csv', 'w') as f:
    hum.to_csv(f)    
hum1 = pd.read_csv('../output/hum.csv')

with open('../output/temp.csv', 'w') as f:
    temp.to_csv(f)    
temp1 = pd.read_csv('../output/temp.csv')

with open('../output/speed.csv', 'w') as f:
    speed.to_csv(f)    
speed1 = pd.read_csv('../output/speed.csv')

with open('../output/cloud.csv', 'w') as f:
    clouds.to_csv(f)    
clouds1 = pd.read_csv('../output/cloud.csv')



df = df.merge(hum1,on='index').merge(temp1,on='index').merge(speed1,on='index').merge(clouds1,on='index')

df.rename(columns={"size": "Number of trips a day"},inplace=True )
    
sns.set(style="ticks", color_codes=True,font_scale = 1.6)

fig1=sns.lmplot(x="wind_speed", y="Number of trips a day",hue="type",data=df)
fig1.savefig("../output/wind_speed.png")

fig2=sns.lmplot(x="temp", y="Number of trips a day", hue="type",data=df)
fig2.savefig("../output/temp.png")

fig3=sns.lmplot(x="humidity", y="Number of trips a day", hue="type",data=df)
fig3.savefig("../output/humidity.png")


sns.set(style="ticks", color_codes=True,font_scale = 1.6)
fig4=sns.lmplot(x="clouds_all", y="Number of trips a day",hue="type",data=df,xlabel='Cloudiness, %')

fig4, ax = plt.subplots()
ax = sns.lmplot(x="clouds_all", y="Number of trips a day", hue="type",data=df)
ax.set( xlabel='Cloudiness, %')
fig4.savefig("../output/cloudiness.png")




# =============================================================================
# pairplot :Plot pairwise relationships 
# =============================================================================
sns.set(font_scale=1.7)
sns.pairplot(df[["wind_speed","temp","humidity","clouds_all"]],diag_kind="kde",plot_kws=dict(s=10))


