import numpy as np
from obspy import read, read_inventory, Trace, Stream
from obspy.signal.rotate import rotate_ne_rt
from obspy.geodetics.base import locations2degrees, degrees2kilometers
import pandas as pd
import numpy as np
import os, warnings
import matplotlib.pyplot as plt
from math import cos, sin, radians
warnings.filterwarnings('ignore')

'''
sigma: damping ratio (%)
w0: natural frequency of SDOF or MDOF oscillator
dt: time interval
x: displacement of the top floor
a: accleration of the  record
j: time increment
'''

def T2w(T):
	return 2*np.pi/T

def def_w0(n_floor):
	return 1/(n_floor*0.1)

def rotate_signals(s1, s2, ang):
	ba = radians(ang)
	r = - s2 * sin(ang) - s1 * cos(ang)
	t = - s2 * cos(ang) + s1 * sin(ang)
	return r, t
def acc2disp(st):
	# Get Displacement Waveform
	st.detrend('linear')
	st.taper(max_percentage=0.05)
	# st.filter('lowpass',freq=1)
	st_disp = st.copy()
	# ACC 2 VEL
	st_disp.integrate('cumtrapz')
	st_disp.detrend('linear')
	st_disp.taper(max_percentage=0.05)
	st_disp.filter('bandpass',freqmin=0.8,freqmax=10)# 0.8 5
	# VEL 2 DISP
	st_disp.integrate()
	st_disp.detrend('linear')
	st_disp.taper(max_percentage=0.05)
	st_disp.filter('bandpass',freqmin=0.8,freqmax=10)# 0.8 5
	return st, st_disp

# Read an earthquake
year = '2024'
evid = '2130'
sta = 'TOLM'
ev_path = f'../DBs/EQData/{year}/{evid}'

# Load Event Catalog
ev_db = pd.read_csv('../DBs/EQCatalog/catalog2021-2024.csv', skiprows=range(1, 2131))
ev_db[['Time']] = ev_db[['Time']].apply(pd.to_datetime)

# Sentinella Building Info
building_db = pd.read_csv('../DBs/sentinella_db.csv',sep=',')
building_db['Angle'].fillna(building_db['GearthAngle'], inplace=True)

# Reverse Channels
reversed_chan = {'CORD':'X','MNFL':'Y','PRPN':'X','UNIU':'N'}#,'UNIU':'E'

path = '../DBs/EQData/'

for i, row in tqdm(ev_db.iterrows()):
	year = str(row['Time'].year)
	evid = str(row['evid'])

	# mseeds = os.listdir(os.path.join(path,year,evid))
	# stas = []
	# for mseed in mseeds:
	# 	sta = mseed.split('.')[0]
	# 	if sta not in stas:
	# 		stas.append(sta)
	# for sta in stas:
	if True:
		st = read(os.path.join(path,year,evid)+'/'+sta+'.*',format='MSEED')
		st.merge()
		st = st.select(channel='HN*')
		st.sort()
		#### ONLY FOR SURICAT !!! ####
		for tr in st:
			tr.data = tr.data * 0.000076 * 9.80665#*10**7

		# Remove Vertical Component
		st.remove(st[-1])
		if len(st) > 3:
			locs = []
			for tr in st:
				if tr.stats.location not in locs:
					locs.append(tr.stats.location)
			sta_build_db = building_db[building_db.Station==sta]
			n_floor = sta_build_db['Number of Floor'].iloc[0]
			f0 = sta_build_db['f0 building [hz]'].iloc[0]
			angle = sta_build_db['Angle'].iloc[0]
			f0s = [float(i) for i in f0.split('-')]
			T0 = round(((f0s[1]-f0s[0])/2)+f0s[0],2)
			sigma = 0.05
			print(sta,n_floor,def_w0(n_floor),f0s,T0,angle)

			# Get Top-Bottom Waveform
			st_bottom = st.select(location=locs[0])
			st_top = st.select(location=locs[1])
			# Reverse Channels
			for tr in st_top:
				if tr.stats.station in list(reversed_chan.keys()):
					if tr.stats.channel[-1] == reversed_chan[tr.stats.station]:
						tr.data = -tr.data

			anguse = angle
			if np.isnan(angle):
				x,y = rotate_signals(st_bottom[1].data, st_bottom[0].data, 0)
				st_bottom[0].data = x
				st_bottom[1].data = y
			else:
				x,y = rotate_signals(st_bottom[1].data, st_bottom[0].data, anguse)
				st_bottom[0].data = x
				st_bottom[1].data = y	

			st_bottom, st_bottom_disp = acc2disp(st_bottom)
			st_top, st_top_disp = acc2disp(st_top)			

			w0 = T2w(T0)
			for tr_bot,tr_bot_disp,tr_top,tr_top_disp in zip(st_bottom,st_bottom_disp,st_top,st_top_disp):
				top_pred_disp = [0,0]
				top_pred_acc = [0,0]
				for i in range(tr_bot.stats.npts):
					if i > 1:
						dt = tr_bot.stats.delta
						# Z-transform parameters
						wd = w0*np.sqrt(1-sigma**2)
						b1 = 2*np.exp(-sigma*w0*dt)*np.cos(wd*dt)
						b2 = -np.exp(-2*sigma*w0*dt)
						S0 = np.exp(-sigma*w0*dt)*np.sin(wd*dt)/(wd*dt)
	
						val_d_i = b1*top_pred_disp[i-1] + b2*top_pred_disp[i-2] - S0*(dt**2)*tr_bot.data[i-1]
						val_a_i = b1*top_pred_acc[i-1] + b2*top_pred_acc[i-2] - S0*(tr_bot.data[i]-2*tr_bot.data[i-1]+tr_bot.data[i-2])
						top_pred_disp.append(val_d_i)
						top_pred_acc.append(val_a_i)
				dt = tr_bot.stats.delta
				tr_top_pred_disp = Trace(data=np.array(top_pred_disp))
				tr_top_pred_disp.stats.delta = dt
				tr_top_pred_disp.filter('bandpass',freqmin=0.8,freqmax=10)
				tr_top_pred_disp.detrend('linear')
				tr_top_pred_disp.taper(0.05)
				
				tr_top_pred_acc = Trace(data=np.array(top_pred_acc))
				tr_top_pred_acc.stats.delta = dt
				tr_top_pred_acc.filter('bandpass',freqmin=0.8,freqmax=10)
				tr_top_pred_acc.detrend('linear')
				tr_top_pred_acc.taper(0.05)

				base_sig = tr_top_pred_acc.data# + tr_bot.data
				disp_meas = tr_top_pred_disp.data + tr_bot_disp.data 

				drift_calc = tr_top_pred_disp.data# - tr_bot_disp.data
				drift_meas = tr_top_disp.data - tr_bot_disp.data

				fig, axs = plt.subplots(2,1,sharex=True,sharey=False,dpi=300,figsize=(12,6))
				axs[0].plot(tr_top_disp.data,color='blue',label=f'{tr_top_disp.stats.channel} displacement')#tr_bot.times(),
				axs[0].plot(disp_meas,color='orange',label=f'{tr_top_disp.stats.channel} displacement simulated')
				axs[1].plot(drift_meas,color='blue',label=f'{tr_top_disp.stats.channel} drift')
				axs[1].plot(drift_calc,color='orange',label=f'{tr_top_disp.stats.channel} drift simulated')
				axs[0].legend(loc='upper right')
				axs[1].legend(loc='upper right')
				axs[0].set_xlim([0,4000])
				axs[0].set_title(f'Evid:{evid} | ID:{tr_top.id} | W0:{T0} | Sigma:{sigma} | Angle:{angle}')
				plt.tight_layout()
				try:
					os.mkdir(f'../Figures/Relative_Displacement/{evid}/')
				except:
					pass
				plt.savefig(f'../Figures/Relative_Displacement/{evid}/{tr_top.id}_{T0}_{sigma}.png')
				plt.close('all')
