# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 23:06:03 2018

@author: li
"""
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None) 

weather = pd.read_csv('../input/weather/weather_local_3_tuple.csv')
weather = pd.DataFrame(data=weather, columns=['dt_iso','weather_main'])
weather['dt_iso'] = pd.to_datetime(weather['dt_iso'])

weather['Hour'] = weather['dt_iso'].dt.hour
weather['Date'] = weather['dt_iso'].dt.date
weather['day_of_week']=weather['dt_iso'].dt.dayofweek
del weather['dt_iso']
weather.Date = weather.Date.astype(str)
weather.Hour = weather.Hour.astype(str)

def write2file(infile):
 #   df = pd.read_csv("../input/bike_2_years/07JourneyDataExtract25May2016-31May2016.csv")
    df = pd.read_csv(infile)
    df['Start Date'] = pd.to_datetime(df['Start Date'],format='%d/%m/%Y %H:%M')  
     
    Number = pd.DataFrame(data=df.groupby([df['Start Date'].dt.date,df['Start Date'].dt.hour]).size(),columns=['size'])
    Number.index.names = ['Date','Hour']
    Number.reset_index(inplace=True)
    with open('../output/size_perHour.csv', 'a') as f:
       Number.to_csv(f, header=False,index=False)

import glob
filenames = glob.glob("../input/bike_2_years/*.csv")
for name in filenames:
    write2file(name)
    
Number = pd.read_csv('../output/size_perHour.csv',names = ['Date','Hour', 'size'])    

Number.Date = Number.Date.astype(str)
Number.Hour = Number.Hour.astype(str)
       
wea_num = pd.merge(Number,weather,on=['Date','Hour'])

bank_holiday=['2016-05-30','2016-08-29','2016-12-21','2016-12-22','2016-12-23',
              '2016-12-26','2016-12-27','2016-12-28','2016-12-29',
              '2016-12-30','2017-01-02','2017-04-14',
              '2017-04-14','2017-04-17','2017-05-01','2017-05-29',
              '2017-08-28','2017-12-21','2017-12-22',
              '2017-12-25','2017-12-26','2017-12-27','2017-12-28','2017-12-29','2018-01-01','2018-01-02',
              '2018-03-30','2018-04-02','2018-05-07','2018-05-28']

wea_num_weekday = wea_num[wea_num['day_of_week']<=4]
workday = wea_num_weekday[~np.isin(wea_num_weekday['Date'], bank_holiday)]

holiday = wea_num[(wea_num['day_of_week']>4)]


# =============================================================================
# 
# =============================================================================
wea_num_weekday = wea_num[wea_num['day_of_week']<=4]
workday = wea_num_weekday[~np.isin(wea_num_weekday['Date'], bank_holiday)]

workday = workday[np.isin(workday["weather_main"],["('Rain',)","('Clear',)","('Snow',)"])]
days = {"('Rain',)":"Rain","('Clear',)":"Clear","('Snow',)":"Snow"}

workday["weather_main"] = workday["weather_main"].apply(lambda x: days[x])
workday.rename(columns={"weather_main": "Weather Type"},inplace=True)

sns.set(font_scale=2)  # crazy big
fig, ax = plt.subplots()
ax = sns.lmplot(x="Hour", y="size",hue="Weather Type",fit_reg=False,
           data=workday,scatter_kws={"s": 33},markers=["o", "x", "1"],
           size=6, aspect=1.4)
ax.set(title='Weekdays', xlabel='Time of day', ylabel='Number of Trips',xticks=[0,4,8,12,16,20,23], ylim=(-200,5600))
plt.show()
# =============================================================================
# 
# =============================================================================
wea_num_weekday = wea_num[wea_num['day_of_week']<=4]
workday = wea_num_weekday[~np.isin(wea_num_weekday['Date'], bank_holiday)]

workday = workday[np.isin(workday["weather_main"],["('Drizzle', 'Rain')","('Mist',)","('Haze',)"])]
days = {"('Drizzle', 'Rain')":"Drizzle,Rain","('Mist',)":"Mist","('Haze',)":"Haze"}

workday["weather_main"] = workday["weather_main"].apply(lambda x: days[x])
workday.rename(columns={"weather_main": "Weather Type"},inplace=True)

sns.set(font_scale=2)  # crazy big
fig, ax = plt.subplots()
ax = sns.lmplot(x="Hour", y="size",hue="Weather Type",fit_reg=False,
           data=workday,scatter_kws={"s": 33},markers=["o", "x", "1"],
           size=6, aspect=1.4)
ax.set(title='Weekdays', xlabel='Time of day', ylabel='Number of Trips',xticks=[0,4,8,12,16,20,23], ylim=(-200,5600))
plt.show()

# =============================================================================
# 
# =============================================================================
holiday = wea_num[(wea_num['day_of_week']>4)]
holiday = holiday[np.isin(holiday["weather_main"],["('Rain',)","('Clear',)","('Snow',)"])]
days = {"('Rain',)":"Rain","('Clear',)":"Clear","('Snow',)":"Snow"}

holiday["weather_main"] = holiday["weather_main"].apply(lambda x: days[x])
holiday.rename(columns={"weather_main": "Weather Type"},inplace=True)

sns.set(font_scale=2)  # crazy big
fig, ax = plt.subplots()
ax = sns.lmplot(x="Hour", y="size",hue="Weather Type",fit_reg=False,
           data=holiday,scatter_kws={"s": 33},markers=["o", "x", "1"],
           size=6, aspect=1.4)
ax.set(title='Weekends', xlabel='Time of day', ylabel='Number of Trips',xticks=[0,4,8,12,16,20,23], ylim=(-200,4600))
plt.show()
# =============================================================================
# 
# =============================================================================
holiday = wea_num[(wea_num['day_of_week']>4)]
holiday = holiday[np.isin(holiday["weather_main"],["('Drizzle', 'Rain')","('Mist',)","('Haze',)"])]
days = {"('Drizzle', 'Rain')":"Drizzle,Rain","('Mist',)":"Mist","('Haze',)":"Haze"}

holiday["weather_main"] = holiday["weather_main"].apply(lambda x: days[x])
holiday.rename(columns={"weather_main": "Weather Type"},inplace=True)

sns.set(font_scale=2)  # crazy big
fig, ax = plt.subplots()
ax = sns.lmplot(x="Hour", y="size",hue="Weather Type",fit_reg=False,
           data=holiday,scatter_kws={"s": 33},markers=["o", "x", "1"],
           size=6, aspect=1.4)
ax.set(title='Weekends', xlabel='Time of day', ylabel='Number of Trips',xticks=[0,4,8,12,16,20,23], ylim=(-200,4600))
plt.show()
# =============================================================================
# 
# =============================================================================






Y= workday['size'].groupby([ workday['weather_main'], workday['Hour']]).median()
Y = Y.unstack(1)
Y.columns=[u'0', u'1', u'10', u'11', u'12', u'13', u'14', u'15', u'16', u'17', u'18', u'19', u'2', u'20', u'21', u'22', u'23', u'3', u'4', u'5', u'6', u'7', u'8', u'9']
Y=Y[[str(i) for i in range(24)]]

Z= holiday['size'].groupby([ holiday['weather_main'], holiday['Hour']]).median()
Z = Z.unstack(1)
Z.columns=[u'0', u'1', u'10', u'11', u'12', u'13', u'14', u'15', u'16', u'17', u'18', u'19', u'2', u'20', u'21', u'22', u'23', u'3', u'4', u'5', u'6', u'7', u'8', u'9']
Z=Z[[str(i) for i in range(24)]]


tem=Y.dropna(axis=0,how='any')
tem['sum'] = tem.apply(sum,axis=1)
tem.sort_values(by='sum',inplace=True)
tem = tem.drop('sum', 1)
tem.iloc[[10,2,9,6],:].T.plot(figsize=(10,9),grid=True,xticks=[0,4,8,12,16,20,23],fontsize=20)
#tem.T.plot(figsize=(12,26))


classify_x1 = Y[['6','7','8','9','10','11']]
Y.iloc[[0,1,71,72],:].T.plot(figsize=(10,9),grid=True,xticks=[0,4,8,12,16,20,23],fontsize=20)

tem.iloc[[1,2,3,4,5,6,7,8],:].T.plot(figsize=(12,26),grid=True)
tem.T.plot(figsize=(12,26))



tem2=Z.dropna(axis=0,how='any')
tem2['sum'] = tem2.apply(sum,axis=1)
tem2.sort_values(by='sum',inplace=True)
tem2 = tem2.drop('sum', 1)
tem2.iloc[[1,2,3,4,5,6,7,8],:].T.plot(figsize=(12,26),grid=True)
tem2.T.plot(figsize=(12,26),grid=True)


Y.iloc[:,:5].plot()

Y.iloc[:,:5].plot()
Y.iloc[:,[1,5,6,7,8]].plot()
Y.iloc[:,:5].plot()
Y.iloc[:,:5].plot()
Y.iloc[:,:5].plot()



