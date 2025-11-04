from funcs import *
from thomson_haskell import *
from ztransform import *
import numpy as np
from obspy import read
import matplotlib.pyplot as plt

# Seismic Signal
PATH = 'WF/1/'
station = 'TOLM'
bot_channel = 'E'
top_channel = 'X'
st = read(f'{PATH}{station}.*.HN{bot_channel}.*',format='MSEED')
st_bottom = st.select(location='00')
st = read(f'{PATH}{station}.*.HN{top_channel}.*',format='MSEED')
st_top = st.select(location='03')
st_bottom, _, st_bottom_disp = acc2disp(st_bottom,freqmin=1,freqmax=40)
st_top, _, st_top_disp = acc2disp(st_top,freqmin=1,freqmax=40)
st_drift = top2drift(st_bottom_disp,st_top_disp)
dt = st_bottom_disp[0].stats.delta
sr = st_bottom_disp[0].stats.sampling_rate
npts = st_bottom_disp[0].stats.npts
# Structural Information
height = 12
f0 = 6.3
zeta = 0.05
Q = zeta2Q(zeta)
vs = 307
w0 = f2w(f0)
rho = 2400
half_space = {'Height':30,'Vs':300,'Q':30,'Density':3000}



'''
Z-Transform 
'''
ztransform_drift = Stream()
ztransform_top_displacement = Stream()
for tr_bot in st_bottom:
  all_drift_pred_dis = []
  drift_pred_dis = [0,0]
  for i in range(tr_bot.stats.npts):
    if i > 1:
        drift_pred_dis = ztransform(i,tr_bot.data,w0,zeta,dt,drift_pred_dis,outtype='DIS')
  ztransform_drift = array2stream(ztransform_drift,drift_pred_dis,dt)
  ztransform_top_displacement = drift2top(st_bottom_disp,ztransform_drift)

'''
Thomson-Haskell
'''
# Thomson Haskell Model Inputs
qs = np.insert([Q],0,half_space['Q'])[::-1]
dn = np.insert([rho],0,half_space['Density'])
vs = np.insert([vs],0,half_space['Vs'])[::-1]
H = [height]

thomson_haskell_drift = Stream()
thomson_haskell_top_displacement = Stream()
for tr_bot in st_bottom_disp:
  fft_bot, freqs = calc_fft(tr_bot.data,tr_bot.stats.npts,tr_bot.stats.sampling_rate)
  # Transfer Function
  tf = transfer_function_n_layer(freqs, H,vs,rho,qs)
  fft_top_pred = fft_bot*tf
  top_pred_disp = calc_ifft(fft_top_pred)
  # Top DISPLACEMENT
  thomson_haskell_top_displacement = array2stream(thomson_haskell_top_displacement,top_pred_disp,dt)
  # Drift
  thomson_haskell_drift = top2drift(st_bottom_disp,thomson_haskell_top_displacement)


# Plotting
# Top Displacement Waveform
fig,axs = plt.subplots(ncols=2,nrows=3,sharex=False,sharey=False,dpi=300,figsize=(16,9))
axs[0][0].plot(st_top_disp[0].times(),st_top_disp[0].data,color='k',label='Observation')
axs[0][0].plot(ztransform_top_displacement[0].times(),ztransform_top_displacement[0].data,color='orange',label='Z-Transform')
axs[0][0].plot(thomson_haskell_top_displacement[0].times(),thomson_haskell_top_displacement[0].data,color='purple',label='Thomson-Haskell')
# Drift Waveform
axs[1][0].plot(st_drift[0].times(),st_drift[0].data,color='k',label='Observation')
axs[1][0].plot(ztransform_drift[0].times(),ztransform_drift[0].data,color='orange',label='Z-Transform')
axs[1][0].plot(thomson_haskell_drift[0].times(),thomson_haskell_drift[0].data,color='purple',label='Thomson-Haskell')
# Bottom Waveform
axs[2][0].plot(st_bottom_disp[0].times(),st_bottom_disp[0].data,color='k',label='Observation')

# Top Displacement Spectrum
amps_top, freqs = calc_absfft(st_top_disp[0].data,npts,sr)
axs[0][1].loglog(freqs,amps_top,color='k',label='Observation')
amps_ztrans, _ = calc_absfft(ztransform_top_displacement[0].data,npts,sr)
axs[0][1].loglog(freqs,amps_ztrans,color='orange',label='Z-Transform')
amps_thomson, _ = calc_absfft(thomson_haskell_top_displacement[0].data,npts,sr)
axs[0][1].loglog(freqs,amps_thomson,color='purple',label='Thomson-Haskell')
# Drift Spectrum
amps_drift, _ = calc_absfft(st_drift[0].data,npts,sr)
axs[1][1].loglog(freqs,amps_drift,color='k',label='Observation')
amps_ztrans, _ = calc_absfft(ztransform_drift[0].data,npts,sr)
axs[1][1].loglog(freqs,amps_ztrans,color='orange',label='Z-Transform')
amps_thomson, _ = calc_absfft(thomson_haskell_drift[0].data,npts,sr)
axs[1][1].loglog(freqs,amps_thomson,color='purple',label='Thomson-Haskell')
# Bottom Spectrum
amps_bot, _ = calc_absfft(st_bottom_disp[0].data,npts,sr)
axs[2][1].loglog(freqs,amps_bot,color='k',label='Observation')

for i, ax in enumerate(axs.flat):
    ax.legend()
    ax.grid('on','both')
    if i % 2 == 0:
      ax.set_xlim([0,50])
    else:
      ax.set_xlim([0.1,20])
plt.show()
