from funcs import *
from thomson_haskell import *
from ztransform import *
import numpy as np
from obspy import read
import matplotlib.pyplot as plt

# Seismic Signal
PATH = 'WF/1/'
station = 'TOLM'
channel = 'E
st = read(f'{PATH}{station}.*.HN{channel}.*',format='MSEED')
st_bottom = st.select(location='00')
st_bottom, _, st_bottom_disp = acc2disp(st_bottom,freqmin=1,freqmax=40)

# Structural Information
f0 = 6.3
zeta = 0.05
Q = zeta2Q(zeta)
vs = 307
w0 = f2w(f0)

'''
Z-Transform 
'''
st_drift_dis = Stream()
for tr_bot in st_bottom:
  dt = tr_bot.stats.delta
  all_drift_pred_dis = []
  drift_pred_dis = [0,0]
  for i in range(tr_bot.stats.npts):
    if i > 1:
        drift_pred_dis = jin2004(i,tr_bot.data,w0,zeta,dt,drift_pred_dis,outtype='DIS')
  st_drift_dis = array2stream(st_drift_dis,drift_pred_dis,dt)
