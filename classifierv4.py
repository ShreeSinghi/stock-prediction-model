# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 19:25:00 2018

@author: Singhi

test results:
    .first layer best: 64
    .second layer best: 32
    .third layer best: 8
    .activator best: leaky relu  
    .optimiser best: adam/nadam
    .concat best layer: after 1

v1: normal
v2: date, stocktype in different layer
v3: time split into y/m/d
v4: replaced 1m open data with 5m candles
    
do scatter plot
"""
tilldate = '2022-10-21'

import numpy as np
from adabound import AdaBound
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import random as python_random

from keras.losses import logcosh
from keras.models import load_model, Model
from keras.layers import Dense, GaussianNoise, Input, Activation, Concatenate
from keras.regularizers import l2
from keras import backend as K
from keras.callbacks import TensorBoard, LambdaCallback
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder

seed = 1

np.random.seed(seed)
python_random.seed(seed)
tf.random.set_seed(seed)

def shuffle(matrixs):
    widths = np.cumsum([0]+[matrix.shape[1] for matrix in matrixs])
    big = np.hstack(matrixs)
    np.random.shuffle(big)
    return [big[:,widths[i]:widths[i+1]].copy() for i in range(len(matrixs))]
    
def scplt():
    plt.scatter(model.predict([x1,x2]), y)
    plt.show()

def important():
    w = abs(model.get_weights()[0]).mean(1)
    plt.plot(w)


def init():
    global  data, idays, fdays, stocks, lens, targ, inlen, data2
    global  bestnacc, bestbacc, besttacc, data2

    idays   = 3   #intra days
    fdays   = 5   #full days
    inlen   = fdays*4 + idays*312 + 5

    stocks  = ['nifty', 'bank']

    data = {}; lens = {}; targ = {};  data2 = {};
    stds = {'nifty':8579., 'bank':20204., 'niftyt':84. , 'bankt':356.}
    means= {'nifty':3829., 'bank':8849.,  'niftyt':113., 'bankt':500.}
    encoder = OneHotEncoder(sparse=False)
    
    for stock in stocks:       

        data[stock] = pd.read_pickle(f'5{stock}.pkl')
        
        data[stock] -= means[stock]
        data[stock] /= stds[stock]

        daily = pd.read_pickle(f'd{stock}.pkl')
        weekdays = array([datetime.fromisoformat(x).weekday() for x in daily.index])
        years = (array([datetime.fromisoformat(x).year for x in daily.index]) - 2016)/3.5
        months = (array([datetime.fromisoformat(x).month for x in daily.index]) - 6.5)/3.5
        days = (array([datetime.fromisoformat(x).day for x in daily.index]) - 15.5)/8.5
        
        daily -= means[stock]
        daily /= stds[stock]
        
        # create day of week list and replace saturdays with fridays
        weekdays[weekdays==5] = 4
        weekdays = encoder.fit_transform(weekdays.reshape(-1,1))
        
        daily = np.vstack(( daily.o, daily.h, daily.l, daily.c )).ravel('F')
        data[stock] = np.vstack((data[stock].o, data[stock].h, data[stock].l, data[stock].c)).ravel('F')

        lens[stock] = len(data[stock])//312 - idays - fdays

        stocktype = ((stock=='nifty') - (stock=='bank'))

        enc1 = array([data[stock][(fdays+i)*312:(fdays+i+idays)*312] for i in range(lens[stock])])
        enc2 = array([daily[i*4:(fdays+i)*4] for i in range(lens[stock])])
                
        data[stock] = np.hstack((
                                 enc1,
                                 enc2,
                                 weekdays[idays+fdays:],
                      ))
        
        data2[stock] = np.hstack((
                                  np.ones((lens[stock], 1)) * stocktype,
                                  years[idays+fdays:, None],    
                                  months[idays+fdays:, None],
                                  days[idays+fdays:, None]
                      ))
        
        daily = pd.read_pickle(f'd{stock}.pkl')
        targ[stock] = (daily.h-daily.l-stds[stock+'t'])/stds[stock+'t']
        targ[stock] = targ[stock][fdays+idays:, None]

def create_model(param):    
    input_layer = Input(shape=(inlen,))
    input_2 = Input(shape=(4,))
    
    x = Dense(64, use_bias=True)(input_layer)
    x = Activation('relu')(x)
    
    x = Concatenate()([x, input_2])
    
    x = Dense(32, use_bias=True)(x)
    x = Activation('relu')(x)
    
    x = Dense(8, use_bias=True)(x)
    x = Activation('relu')(x)

    x = Dense(1, use_bias=True)(x)

    
    model = Model([input_layer, input_2], x)
    
    
    model.compile(
        optimizer = 'adam',
        loss = 'mean_squared_logarithmic_error'
    )
    
    return model

if __name__ == '__main__':
    init()

    np.random.seed(seed)
    python_random.seed(seed)
    tf.random.set_seed(seed)
    
    x1 = np.vstack([data[stock] for stock in stocks])
    x2 = np.vstack([data2[stock] for stock in stocks])
    y  = np.vstack([targ[stock] for stock in stocks])
    x1, x2, y = shuffle((x1,x2,y))
    
    np.random.seed(seed)
    python_random.seed(seed)
    tf.random.set_seed(seed)
    
    model = create_model(0)
    candles = model.fit([x1, x2], y, epochs=256, batch_size=32,
                        validation_split=0.1, use_multiprocessing=True)
