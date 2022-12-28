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

also if duplicates are found, values from old file are replaced
"""

import pandas as pd
import os
    
nfiles = ['NIFTYnifty.csv.pkl']

bfiles = ['BANKNIFTYbank.csv.pkl']

nfiles += ['intranifty.pkl']
bfiles += ['intrabank.pkl']

ndfs = []

for nfile in nfiles:
    try:
        currdf = pd.read_pickle(nfile)
        ndfs.append(currdf)
    except Exception:
        pass

ndf = pd.concat(ndfs)


bdfs = []
for bfile in bfiles:
    try:
        currdf = pd.read_pickle(bfile)
        bdfs.append(currdf)
    except Exception:
        pass

bdf = pd.concat(bdfs)

ndf = ndf[~ndf.index.duplicated(keep='first')]
bdf = bdf[~bdf.index.duplicated(keep='first')]

ndf.sort_index(inplace=True)
bdf.sort_index(inplace=True)


ndf.to_pickle('intranifty.pkl')
bdf.to_pickle('intrabank.pkl')

# for day in set(x[0] for x in ndf.index):
    # pass
