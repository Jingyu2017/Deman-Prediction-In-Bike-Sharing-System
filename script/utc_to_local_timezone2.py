#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
different from version 1:
without description
with cloud_all
"""

from datetime import datetime
from pytz import timezone
import pandas as pd
weather = pd.read_csv('../input/weather/weather_16_18.csv')
weather = pd.DataFrame(data=weather, columns=['dt_iso','temp',
                                              'humidity','wind_speed',
                                              'clouds_all'
                                              ])
weather.drop_duplicates(inplace=True)
weather['dt_iso'] = weather['dt_iso'].map(lambda x: x.split(" +0000")[0])
weather['dt_iso'] = pd.to_datetime(weather['dt_iso'])

bst = weather[ 
        ((weather['dt_iso']>datetime(2016, 3, 27, 0, 59, 0))&
       (weather['dt_iso']<datetime(2016, 10, 30, 0, 59, 0)) )|
       ( (weather['dt_iso']>datetime(2017, 3, 26, 0, 59, 0))&
       (weather['dt_iso']<datetime(2017, 10, 29, 0, 59, 0)) )|
       (weather['dt_iso']>datetime(2018, 3, 25, 0, 59, 0))
       ]

utc = weather[  ((weather['dt_iso']>datetime(2016, 10, 29, 0, 59, 0))&
              (weather['dt_iso']<datetime(2017, 3, 26, 0, 59, 0))) |
               ((weather['dt_iso']>datetime(2016, 10, 29, 0, 59, 0))&
              (weather['dt_iso']<datetime(2018, 3, 25, 0, 59, 0)))
              ]

bst['dt_iso'] = bst['dt_iso'].map(lambda x: timezone('UTC').localize(x))
fmt = "%Y-%m-%d %H:%M:%S"
bst['dt_iso']= bst['dt_iso'].map(lambda x: x.astimezone(timezone('Europe/London')).strftime(fmt)) 
new = pd.concat([bst,utc])



with open('../input/weather/weather_local_2.csv','w') as f:
    new.to_csv(f, header=True)
