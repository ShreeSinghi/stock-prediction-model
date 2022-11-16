# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 09:53:25 2019

@author: Singhi

ideas:
    .use only high low candles for intraday (1m/5m)
    
    .try to predict open/close given high and low
    
    .predict range of next day
    
    .drop either close or open column for intraday coz one trade difference
    
    .either use 5m candles or 1m open data
    
    .mnist input image -> bottleneck -> recreate image
    
    .feed in date, month, year (for newsly stuff)

"""

from nsepy import get_history
from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd
from keras.models import load_model
from matplotlib.pyplot import plot, show, legend, hist, fill
from keras.losses import logcosh
from numpy import array
from keras import backend as K
from adabound import AdaBound
import classifier
import tensorflow as tf
import yfinance as yf

tf.compat.v1.disable_eager_execution()
load = lambda x: load_model(x, {'AdaBound':AdaBound, 'func':logcosh})

def binary(input_tensor):
    def func(y_true, y_pred):
        return K.mean(K.square(y_true) *                                      \
                      (K.square(K.sign(y_true) - y_pred)) *                   \
                      (.5 + K.square(input_tensor[:, -1]+2.8)), -1)
    return func

def new_pred(stonk):
    global days, data, predictor, csvs, omeans, ostds
    
    if stonk == 'nifty':
        data = yf.download("^NSEI", period="1d", interval="1d", progress=False).iloc[0]
    else:
        data = yf.download("^NSEI", period="1d", interval="1d", progress=False).iloc[0]
    
    o, c, h, l = (data['Open']-omeans[stonk])/omeans[stonk],                  \
                 (data['Close']-cmeans[stonk])/cmeans[stonk],                 \
                 (data['High']-hmeans[stonk])/hmeans[stonk],                  \
                 (data['Low']-lmeans[stonk])/lmeans[stonk]
    
    df = csvs[stonk]
    
    ind = len(df) + 1

    enc = np.vstack(( (df.Open [ind-days:ind-1] - omeans[stonk]) / ostds[stonk],
                      (df.Close[ind-days:ind-1] - cmeans[stonk]) / cstds[stonk],
                      (df.High [ind-days:ind-1] - hmeans[stonk]) / hstds[stonk],
                      (df.Low  [ind-days:ind-1] - lmeans[stonk]) / lstds[stonk] )).ravel('F')
    
    enc = np.append(enc, (o, c, h, l))
    
    enc = np.append(enc,
                   ((stonk=='nifty')-(stonk=='banknifty'), (ind-days)/1000 - 2.8)
                   )[None, :]

    res = float(predictor.predict(enc))
    chances = 50 + abs(res)*50
    
    print(f'Predicted {round(chances, 1)}% chance of {"green" if res>0 else "red"}')
    

def train_models():
    global modelname, predictor
    
    print('Training...')
    classifier.module_function()
    print('Done')
    niftymodel = load('nbest.h5')
    bankmodel  = load('bbest.h5')
    
    if modelname == 'nifty':
        predictor = niftymodel
    else:
        predictor = bankmodel

def switch_model():
    """
    Switches model between bankmodel and niftymodel
    """
    global modelname, predictor, niftymodel, bankmodel, csvs
    if modelname=='nifty':
        modelname = 'banknifty'
        predictor = bankmodel
    elif modelname=='banknifty':
        modelname ='nifty'
        predictor = niftymodel
    print('Switched to ' + modelname)

def predict_on(stonk, date):
    """
    Prints prediction on 'date' given its 'openval'
    """
    global days, data, predictor, csvs, omeans, ostds
    df = csvs[stonk]
    
    ind = df.Date[df.Date==date].index[0] + 1

    enc = np.vstack(( (df.Open [ind-days:ind] - omeans[stonk]) / ostds[stonk],
                      (df.Close[ind-days:ind] - cmeans[stonk]) / cstds[stonk],
                      (df.High [ind-days:ind] - hmeans[stonk]) / hstds[stonk],
                      (df.Low  [ind-days:ind] - lmeans[stonk]) / lstds[stonk] )).ravel('F')

    enc = np.append(enc,
                   ((stonk=='nifty')-(stonk=='banknifty'), (ind-days)/1000 - 2.8)
                   )[None, :]

    res = float(predictor.predict(enc))
    chances = 50 + abs(res)*50
    
    print(f'Predicted {round(chances, 1)}% chance of {"green" if res>0 else "red"}')

def validate_on(stonk, date):
    """
    Prints models accuracy on 'date'
    """
    global days, profits, data, predictor, csvs

    df = csvs[stonk]
    ind = df.Date[df.Date==date].index[0] - days + 1
    
    res = float(predictor.predict(data[stonk][ind][None]))

    chances = 50 + abs(res)*50
    valid = profits[stonk][ind]
    
    print(f'Predicted {round(chances, 1)}% chance of {"green" if res>0 else "red"}')
    print(f'Actual {"green" if valid>0 else "red"}')
    print(f'{"Profit" if res*valid>0 else "Loss"} of {abs(profits[stonk][ind])}')

def capital(stonk, start, end):
    """
    Graphs capital if bot started trading on 'start' and stopped on 'end' and prints stats
    """
    global days, profits, data, predictor, csvs
    df = csvs[stonk]
    
    startind = df.Date[df.Date==start].index[0] - days + 1
    endind   = df.Date[df.Date==end  ].index[0] - days + 1
    v = np.arange(startind, endind+1)

    pred = predictor.predict(data[stonk][v]).flatten()

    foo = np.sign(pred)*profits[stonk][v]
    
    money  = np.cumsum(np.append([0], foo))
    target = np.cumsum(np.append([0], abs(profits[stonk][v])))
    
    plot(money, label = stonk+' Money')
    plot(target, label = stonk+' Target')

    legend()
    show()
    
    predprint = ', '.join(['Sell' if i<0 else 'Buy' for i in pred])
    
    print(f'Predictions were: {predprint}')
    print(f'Final profit: {money[-1]}')
    print(f'Best possible: {target[-1]}')
    print(f'Money Accuracy: {money[-1]/target[-1]*100}%')
    print(f'Loss money: {-round(sum(foo[foo<0]))}')
    print(f'Profit money: {round(sum(foo[foo>0]))}')
    print(f'Ratio: {-sum(foo[foo>0])/sum(foo[foo<0])}')

def valid_sim(stonk):
    """
    Graphs capital on all validation dates from 2005 June 9
    """
    global profits, data, predictor
    pred = predictor.predict(data[stonk]).flatten()

    foo = np.sign(pred)*profits[stonk]
    
    money  = np.cumsum(np.append([0], foo))
    target = np.cumsum(np.append([0], abs(profits[stonk])))
    
    plot(money, label = stonk+' Money')
    plot(target, label = stonk+' Target')

    legend()
    show()
    
    print(f'Final profit: {money[-1]}')
    print(f'Best possible: {target[-1]}')
    print(f'Money Accuracy: {money[-1]/target[-1]*100}%')
    print(f'Loss money: {-round(sum(foo[foo<0]))}')
    print(f'Profit money: {round(sum(foo[foo>0]))}')
    print(f'Ratio: {-sum(foo[foo>0])/sum(foo[foo<0])}')

def init():
    """
    Initializes variables and fetches standard deviations & means for dataset
    """
    global days, stocks, predictor, modelname, niftymodel, bankmodel
    global omeans, hmeans, lmeans, cmeans, ostds, hstds, lstds, cstds
    global nbacc, bbacc, tbacc

    omeans = {}; hmeans = {}; lmeans = {}; cmeans = {};
    ostds = {}; hstds = {}; lstds = {}; cstds = {};
    
    days = 225
    niftymodel = load('nbest.h5')
    bankmodel  = load('bbest.h5')

    modelname = 'nifty'
    predictor = niftymodel
    
    stocks  = ['nifty', 'banknifty']
    
    with open('data/stats.txt') as file:
        f = file.readlines()
        for line in f:
            temp = line.split()
            stock = temp.pop(0)
            if stock not in stocks: continue
            
            ostds[stock],  cstds[stock],  hstds[stock],  lstds[stock]  =  map(float, temp[:4])
            omeans[stock], cmeans[stock], hmeans[stock], lmeans[stock] =  map(float, temp[4:])

def load_data():
    """
    Loads and prepares data from csvs, normalizes it.
    Creates data to feed predictor,
    stores tables, stock lengths, validation days, training days, profits
    """
    global data, days, profits, stocks, lens, csvs, lastdate
    global omeans, hmeans, lmeans, cmeans, ostds, hstds, lstds, cstds
    
    data = {}; lens = {}; profits = {}; csvs = {};
    
    for stock in stocks:
        
        df = pd.read_csv(f'data/{stock}.csv')
        csvs[stock] = df.copy()
        
        data[stock] = np.vstack(( (df.Open  - omeans[stock]) / ostds[stock],
                                  (df.Close - cmeans[stock]) / cstds[stock],
                                  (df.High  - hmeans[stock]) / hstds[stock],
                                  (df.Low   - lmeans[stock]) / lstds[stock] )).ravel('F')

        profits[stock] = (array(df.Open[1:]) - array(df.Close[:-1]))[days-1:]
        lens[stock] = len(profits[stock])

        stocktype = ((stock=='nifty') - (stock=='banknifty'))
        enc = array([data[stock][4*i:4*(i+days)] for i in range(lens[stock])])
        
        data[stock] = np.hstack(( enc,
                                  np.ones((lens[stock], 1)) * stocktype,
                                  np.arange(lens[stock])[:, None]/1000 - 2.8))
    lastdate = df.Date.iloc[-1]
    
    print('Last date: ', df.Date.iloc[-1] )

def refresh_data(d=False):
    """
    Fetches data from server and stores in csvs
    """
    global stocks, indexes, frame
    
    if d:
        enddate = datetime.strptime(d,'%Y-%m-%d').date()
    else:
        enddate = date.today()
    
    for stock in stocks: 
        frame = get_history(symbol=stock.upper(),
                            start=date(2005,6,9),
                            end=enddate,
                            index=True)[['Open','Close', 'High', 'Low']].dropna()
        frame.to_csv(f'data/{stock}.csv')
    
    load_data()