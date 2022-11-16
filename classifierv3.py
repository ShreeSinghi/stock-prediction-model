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
from keras.layers import Dense, GaussianNoise, BatchNormalization, Input, Activation, Concatenate
from keras.regularizers import l2
from keras import backend as K
from keras.callbacks import TensorBoard, LambdaCallback
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder

np.random.seed(2)
python_random.seed(2)
tf.random.set_seed(2)

def shuffle(matrixs):
    widths = np.cumsum([0]+[matrix.shape[1] for matrix in matrixs])
    big = np.hstack(matrixs)
    np.random.shuffle(big)
    return [big[:,widths[i]:widths[i+1]].copy() for i in range(len(matrixs))]
    

def module_function():
    global bestnacc, bestbacc, besttacc, nifty, bank
    init()
    
    my_load_model('nbest.h5')
    bestnacc = nplot('nifty', False)[-1]
    
    my_load_model('bbest.h5')
    bestbacc = nplot('banknifty', False)[-1]
    
    my_load_model('tbest.h5')
    besttacc = nplot('banknifty', False)[-1] + nplot('nifty', False)[-1]

    nifty = np.cumsum(abs(profits['nifty']))/maxprofs['nifty']
    bank  = np.cumsum(abs(profits['banknifty']))/maxprofs['banknifty']

    genobj = generator(256)
    
    my_load_model('nbest.h5')
    model.fit_generator(genobj, epochs=128, steps_per_epoch=64, verbose=0,
                        callbacks=[LambdaCallback(None, save)]) #5 mins
    
    my_load_model('bbest.h5')
    model.fit_generator(genobj, epochs=128, steps_per_epoch=128, verbose=0,
                        callbacks=[LambdaCallback(None, save)]) #5 mins
    
def my_save_model(fname):
    global model
    model.save(fname)

def my_load_model(fname):
    global model
    model = load_model(fname, {'AdaBound': AdaBound, 'func':logcosh})
    model.compile(
        optimizer = model.optimizer,
        loss = binary(model.inputs[0])
    )

def binary(input_tensor):
    def func(y_true, y_pred):
        return K.mean(K.square(y_true) *                                      \
                      (K.square(K.sign(y_true) - y_pred)) *                   \
                      (.5 + K.square(input_tensor[:, -1]+2.8)), -1)
    return func

def money(y_true, y_pred):
    return K.sum(K.sum(K.sign(y_pred)*y_true, -1), -1)

def validsim(stonk):
    global days, y, profits, data, model
    pred = np.sign(model.predict(data[stonk])).flatten()
    
    money  = np.cumsum(pred*profits[stonk])
    target = np.cumsum(abs(profits[stonk]))
    
    plt.plot(money, label = 'Money')
    plt.plot(target, label = 'Target')

    plt.legend()
    plt.show()
    
    print(f'Accuracy: {money[-1]/target[-1]*100}%')

def important():
    w = abs(model.get_weights()[0]).mean(1)
    plt.plot(w)

def nplot(stonk, say=True):
    global data, maxprofs

    if say:
        acc = np.sum(np.sign(model.predict(data[stonk]).flatten())*profits[stonk])/maxprofs[stonk]
        print(f'{stonk} Accuracy: {acc*100}%')
    else:
        acc = np.cumsum(np.sign(model.predict(data[stonk]).flatten())*profits[stonk])/maxprofs[stonk]
    return acc

def init():
    global  data, idays, fdays, stocks, lens, targ, inlen, data2
    global  bestnacc, bestbacc, besttacc, data2

    idays   = 3   #intra days
    fdays   = 5   #full days
    inlen   = fdays*4 + idays*390 + 5

    stocks  = ['nifty', 'bank']

    data = {}; lens = {}; targ = {};  data2 = {};
    stds = {'nifty':8579., 'bank':20204., 'niftyt':84. , 'bankt':356.}
    means= {'nifty':3829., 'bank':8849.,  'niftyt':113., 'bankt':500.}
    encoder = OneHotEncoder(sparse=False)
    
    for stock in stocks:       

        data[stock] = pd.read_pickle(f'intra{stock}.pkl')[:-390]
        
        data[stock] -= means[stock]
        data[stock] /= stds[stock]

        daily = pd.read_pickle(f'd{stock}.pkl')
        weekdays = array([datetime.fromisoformat(x).weekday() for x in daily.index])
        years = (array([datetime.fromisoformat(x).year for x in daily.index]) - 2016)/3.5
        months = (array([datetime.fromisoformat(x).month for x in daily.index]) - 6.5)/3.5
        days = (array([datetime.fromisoformat(x).day for x in daily.index]) - 15.5)/8.5
        
        # to not include last day ofc
        daily = daily[:-1]
        daily -= means[stock]
        daily /= stds[stock]
        
        # create day of week list and replace saturdays with fridays
        weekdays[weekdays==5] = 4
        weekdays = encoder.fit_transform(weekdays.reshape(-1,1))
        
        daily = np.vstack(( daily.o, daily.h, daily.l, daily.c )).ravel('F') 

        lens[stock] = len(data[stock])//390 - idays - fdays + 1

        stocktype = ((stock=='nifty') - (stock=='bank'))

        enc1 = array([data[stock].o[(fdays+i)*390:(fdays+i+idays)*390] for i in range(lens[stock])])
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

def save(x, y):
    global bestnacc, bestbacc, besttacc, bank, nifty

    bacc = nplot('banknifty', False)
    nacc = nplot('nifty', False)

    with open('money.txt', 'w') as f:
        f.write(f"{' '.join(map(str, list( nacc*100  )))}\n"
                f"{' '.join(map(str, list( bacc*100  )))}\n"
                f"{' '.join(map(str, list( nifty*100 )))}\n"
                f"{' '.join(map(str, list( bank*100  )))}")

    if bacc[-1]>bestbacc:
        my_save_model('bbest.h5')
        bestbacc = bacc[-1]

    if nacc[-1]>bestnacc:
        my_save_model('nbest.h5')
        bestnacc = nacc[-1]

    if nacc[-1]+bacc[-1]>besttacc:
        my_save_model('tbest.h5')
        besttacc = nacc[-1]+bacc[-1]

def create_model(loss):    
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
    
    # gradient = AdaBound(lr=5e-5, amsbound=True, final_lr=0.5)
    
    model.compile(
        optimizer = 'adam',
        loss = 'mean_squared_logarithmic_error'
    )
    
    return model

if __name__ == '__main__':
    init()

    np.random.seed(2)
    python_random.seed(2)
    tf.random.set_seed(2)
    
    x1 = np.vstack([data[stock] for stock in stocks])
    x2 = np.vstack([data2[stock] for stock in stocks])
    y  = np.vstack([targ[stock] for stock in stocks])
    x1, x2, y = shuffle((x1,x2,y))
    
    np.random.seed(2)
    python_random.seed(2)
    tf.random.set_seed(2)
    
    model = create_model(0)
    opens = model.fit([x1, x2], y, epochs=256, batch_size=32,
                        validation_split=0.1, use_multiprocessing=True)
    # plt.plot(results.history['loss'], label='loss_0')
    # plt.plot(results.history['val_loss'], label='val_0')

    # plt.legend(loc='upper right')
    # plt.show()
    # callbacks=[LambdaCallback(None, save)])