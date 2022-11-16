# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 22:43:43 2019

@author: Singhi
"""

print('Loading...')

from master import *

init()
refresh_data()

input('Please check if date is correct and press enter')
print('\nSwitched to nifty\n')

predict_on('nifty', 'foo')

print()
switch_model()
print()

predict_on('banknifty', 'foo')

while True: pass

