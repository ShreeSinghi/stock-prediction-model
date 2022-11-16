# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 13:17:45 2022

@author: Shree
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 11:52:37 2022

@author: Shree

need to remove diwali from days list in ohlc daily data else mismatch
"""

import pandas as pd
import os
    
nfiles = ['Intraday 1 Min Data\Consolidated\\NIFTY\\' + i for i in os.listdir('Intraday 1 Min Data\\Consolidated\\NIFTY')]

bfiles = ['Intraday 1 Min Data\Consolidated\\BNF\\' + i for i in os.listdir('Intraday 1 Min Data\\Consolidated\\BNF')]

ndfs = []

for nfile in nfiles:
    try:
        currdf = pd.read_pickle(nfile)
        ndfs.append(currdf)
    except Exception:
        pass

ndf = pd.concat(ndfs)
ndf.sort_index(inplace=True)

bdfs = []
for bfile in bfiles:
    try:
        currdf = pd.read_pickle(bfile)
        bdfs.append(currdf)
    except Exception:
        pass

bdf = pd.concat(bdfs)
bdf.sort_index(inplace=True)

# for day in set(x[0] for x in ndf.index):
    # pass