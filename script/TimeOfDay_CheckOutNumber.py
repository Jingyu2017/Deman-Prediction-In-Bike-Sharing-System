# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 19:40:33 2018

@author: li
"""

# =============================================================================
# to generate statistical data for plotting
# =============================================================================


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)  

def stats(infile):
    df = pd.read_csv(infile)
    df['Start Date'] = pd.to_datetime(df['Start Date'],format='%d/%m/%Y %H:%M')

    
    series = df.groupby([df['Start Date'].dt.date,df['Start Date'].dt.hour]).size().unstack()
    
    with open('TimeOfDay.csv', 'a') as f:
        series.to_csv(f, header=False)
    
    
import glob
filenames = glob.glob("../input/bike_2_years/*.csv")
for name in filenames:
    stats(name)

# =============================================================================
# to plot
# =============================================================================
bank_holiday=['2016-05-30','2016-08-29','2016-12-21','2016-12-22','2016-12-23',
              '2016-12-26','2016-12-27','2016-12-28','2016-12-29',
              '2016-12-30','2017-01-02','2017-04-14',
              '2017-04-14','2017-04-17','2017-05-01','2017-05-29',
              '2017-08-28','2017-12-21','2017-12-22',
              '2017-12-25','2017-12-26','2017-12-27','2017-12-28','2017-12-29','2018-01-01','2018-01-02',
              '2018-03-30','2018-04-02','2018-05-07','2018-05-28']

headers=[i for i in range(24)]
TimeOfDay = pd.read_csv('TimeOfDay.csv',names = headers)
TimeOfDay = TimeOfDay[~np.isin(TimeOfDay.index, bank_holiday)] 
TimeOfDay.index = pd.to_datetime(TimeOfDay.index)
TimeOfDay['day_of_week'] = TimeOfDay.index.to_series().dt.dayofweek



days = {0:'Mon',1:'Tues',2:'Wed',3:'Thurs',4:'Fri',5:'Sat',6:'Sun'}

TimeOfDay['day_of_week'] = TimeOfDay['day_of_week'].apply(lambda x: days[x])



result = TimeOfDay.groupby(TimeOfDay['day_of_week']).mean() 
result = result.T
result.plot()
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xticks(range(0,25,4))
plt.xlabel('Time of day')
plt.ylabel('Number of trips')
plt.show()







