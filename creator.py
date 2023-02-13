# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 16:54:25 2020

@author: Shree
"""


from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from time import time, sleep
from datetime import datetime, date
import yfinance as yf

def clean(df):
    global dates, times
    times = ['09:01', '09:02', '09:03', '09:04', '09:05', '09:06', '09:07', '09:08', '09:09', '09:10', '09:11', '09:12', '09:13', '09:14', '09:15', '09:16', '09:17', '09:18', '09:19', '09:20', '09:21', '09:22', '09:23', '09:24', '09:25', '09:26', '09:27', '09:28', '09:29', '09:30', '09:31', '09:32', '09:33', '09:34', '09:35', '09:36', '09:37', '09:38', '09:39', '09:40', '09:41', '09:42', '09:43', '09:44', '09:45', '09:46', '09:47', '09:48', '09:49', '09:50', '09:51', '09:52', '09:53', '09:54', '09:55', '09:56', '09:57', '09:58', '09:59', '10:00', '10:01', '10:02', '10:03', '10:04', '10:05', '10:06', '10:07', '10:08', '10:09', '10:10', '10:11', '10:12', '10:13', '10:14', '10:15', '10:16', '10:17', '10:18', '10:19', '10:20', '10:21', '10:22', '10:23', '10:24', '10:25', '10:26', '10:27', '10:28', '10:29', '10:30', '10:31', '10:32', '10:33', '10:34', '10:35', '10:36', '10:37', '10:38', '10:39', '10:40', '10:41', '10:42', '10:43', '10:44', '10:45', '10:46', '10:47', '10:48', '10:49', '10:50', '10:51', '10:52', '10:53', '10:54', '10:55', '10:56', '10:57', '10:58', '10:59', '11:00', '11:01', '11:02', '11:03', '11:04', '11:05', '11:06', '11:07', '11:08', '11:09', '11:10', '11:11', '11:12', '11:13', '11:14', '11:15', '11:16', '11:17', '11:18', '11:19', '11:20', '11:21', '11:22', '11:23', '11:24', '11:25', '11:26', '11:27', '11:28', '11:29', '11:30', '11:31', '11:32', '11:33', '11:34', '11:35', '11:36', '11:37', '11:38', '11:39', '11:40', '11:41', '11:42', '11:43', '11:44', '11:45', '11:46', '11:47', '11:48', '11:49', '11:50', '11:51', '11:52', '11:53', '11:54', '11:55', '11:56', '11:57', '11:58', '11:59', '12:00', '12:01', '12:02', '12:03', '12:04', '12:05', '12:06', '12:07', '12:08', '12:09', '12:10', '12:11', '12:12', '12:13', '12:14', '12:15', '12:16', '12:17', '12:18', '12:19', '12:20', '12:21', '12:22', '12:23', '12:24', '12:25', '12:26', '12:27', '12:28', '12:29', '12:30', '12:31', '12:32', '12:33', '12:34', '12:35', '12:36', '12:37', '12:38', '12:39', '12:40', '12:41', '12:42', '12:43', '12:44', '12:45', '12:46', '12:47', '12:48', '12:49', '12:50', '12:51', '12:52', '12:53', '12:54', '12:55', '12:56', '12:57', '12:58', '12:59', '13:00', '13:01', '13:02', '13:03', '13:04', '13:05', '13:06', '13:07', '13:08', '13:09', '13:10', '13:11', '13:12', '13:13', '13:14', '13:15', '13:16', '13:17', '13:18', '13:19', '13:20', '13:21', '13:22', '13:23', '13:24', '13:25', '13:26', '13:27', '13:28', '13:29', '13:30', '13:31', '13:32', '13:33', '13:34', '13:35', '13:36', '13:37', '13:38', '13:39', '13:40', '13:41', '13:42', '13:43', '13:44', '13:45', '13:46', '13:47', '13:48', '13:49', '13:50', '13:51', '13:52', '13:53', '13:54', '13:55', '13:56', '13:57', '13:58', '13:59', '14:00', '14:01', '14:02', '14:03', '14:04', '14:05', '14:06', '14:07', '14:08', '14:09', '14:10', '14:11', '14:12', '14:13', '14:14', '14:15', '14:16', '14:17', '14:18', '14:19', '14:20', '14:21', '14:22', '14:23', '14:24', '14:25', '14:26', '14:27', '14:28', '14:29', '14:30', '14:31', '14:32', '14:33', '14:34', '14:35', '14:36', '14:37', '14:38', '14:39', '14:40', '14:41', '14:42', '14:43', '14:44', '14:45', '14:46', '14:47', '14:48', '14:49', '14:50', '14:51', '14:52', '14:53', '14:54', '14:55', '14:56', '14:57', '14:58', '14:59', '15:00', '15:01', '15:02', '15:03', '15:04', '15:05', '15:06', '15:07', '15:08', '15:09', '15:10', '15:11', '15:12', '15:13', '15:14', '15:15', '15:16', '15:17', '15:18', '15:19', '15:20', '15:21', '15:22', '15:23', '15:24', '15:25', '15:26', '15:27', '15:28', '15:29', '15:30']
    dates = set([x[0] for x in df.index])
    
    for date in dates:
        ftime = df.loc[date].index[0]
        try:
            find = times.index(ftime)
        except ValueError as e:
            print(e, date)
            df.drop(date, level='date')
            continue
            
        val = float(df.loc[(date,ftime)]['o'])
    
        for i in range(find):
            df.loc[(date, times[i]), :] = val
        
        for i in range(find, len(times)):
            try:
                val = float(df.loc[(date, times[i]), :]['c'])
            except KeyError:
                df.loc[(date, times[i]),:] = val
        
        df.sort_index(inplace=True)
        lent = len(df.loc[date])
        if lent > 390:
            df.drop(list(zip([date]*(lent-390), df.loc[date].index[:-390])), inplace=True)
        
    temp = times*(len(df)//390)
    df['time'] = temp
    df = df.set_index('time', append=True).droplevel(2)
    df.sort_index(inplace=True)

    return df

def load_data():
    global stocks, data
    data = pd.read_pickle('pickles/newmain.pkl')
    stocks = data.index.levels[0]
    print('Loaded data')

def fetch_data():
    global data, now
    for stock in ['nifty', 'bank']:
        
        symbol = "^NSEI" if stock=='nifty' else "^NSEBANK"
        df = yf.download(symbol, period="max", interval="1m")
        
        df = df.rename(columns={'Open':'o',
                                'High':'h',
                                'Low':'l',
                                'Adj Close':'c'
            }).drop(columns='Volume').drop(columns='Close')
        
        dates = [str(x) for x in df.index.date]
        times = [str(x)[:5] for x in df.index.time]
        
        index = pd.MultiIndex.from_tuples(list(zip(dates, times)), names=["date", "time"])
        df = df.reset_index().drop('Datetime', axis=1).set_index(index)
        df = clean(df)
        df = pd.concat({stock: df}, names=['stock'])
                
        data = pd.concat([data, df]).sort_index()
        data = data[~data.index.duplicated(keep='last')]

        # data.to_pickle(f'intra{stock}.pkl')
        
    df.sort_index(inplace=True)
    print('Fetched data')

# creates 5m candle data
def create5(df):
    if len(df)%390 != 0:
        raise ValueError
        
    n = len(df)//5
    df2 = df[::5].copy()
    for i in range(n):
        df2[i:i+1] = df.o[i*5], max(df.h[i*5:i*5+5]), min(df.l[i*5:i*5+5]),df.c[i*5+4]
        if not i%10000: print(i,'/',n)
    return df2

# creates 1d candle data
def createday(df):
    if len(df)%390 != 0:
        raise ValueError
        
    n = len(df)//390
    df2 = df[::390].copy()
    for i in range(n):
        df2[i:i+1] = df.o[i*390], max(df.h[i*390:i*390+390]), min(df.l[i*390:i*390+390]),df.c[i*390+389]
        if not i%1000: print(i,'/',n)
    
    df2['date'] = [x[0] for x in df2.index]
    df2.set_index('date', drop=True, inplace=True)
    return df2

load_data()
fetch_data()

# for stock in stocks:
#     pd.to_pickle(createday(data.loc[stock]), f'pickles/d{stock}.pkl')
#     pd.to_pickle(create5(data.loc[stock]), f'pickles/5{stock}.pkl')

# oneday = 24*60*60

# now = time() + hrToSec(5.5)
# nowtime = getTime(now)
# sleep((hrToSec(16)-nowtime+oneday)%oneday)

# while True:
#     fetch_data()
#     sleep(oneday)
