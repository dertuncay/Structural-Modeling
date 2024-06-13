import numpy as np
from obspy import read, read_inventory, Trace, Stream
from obspy.signal.rotate import rotate_ne_rt
import pandas as pd
import numpy as np
import os, warnings
import matplotlib.pyplot as plt
from tqdm import tqdm
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

def GetBackazimuth(azimuth):
	if (azimuth < 0):
		azimuth += 360
		
	if (azimuth <= 180):
		return (180+azimuth)
	else:
		return (azimuth-180)

def T2w(T):
	return 2*np.pi/T

def rotate_signals(s1, s2, ang):
	ba = radians(ang)
	r = - s2 * sin(ang) - s1 * cos(ang)
	t = - s2 * cos(ang) + s1 * sin(ang)
	return r, t

def acc2disp(st,freqmin=0.8,freqmax=10):
	# Get Velocity & Displacement Waveform
	st.detrend('linear')
	st.taper(max_percentage=0.05)
	st.filter('bandpass',freqmin=freqmin,freqmax=freqmax)
	st_disp = st.copy()
	# ACC 2 VEL
	st_vel.integrate('cumtrapz')
	st_vel.detrend('linear')
	st_vel.taper(max_percentage=0.05)
	st_vel.filter('bandpass',freqmin=freqmin,freqmax=freqmax)# 0.8 5
	st_disp = st_vel.copy()
	# VEL 2 DISP
	st_disp.integrate()
	st_disp.detrend('linear')
	st_disp.taper(max_percentage=0.05)
	st_disp.filter('bandpass',freqmin=freqmin,freqmax=freqmax)# 0.8 5
	return st, st_vel, st_disp

def jin2004(tr_bot,w0,sigma,dt,outlist,outtype='DIS'):
	# Z-transform parameters
	wd = w0*np.sqrt(1-sigma**2)
	b1 = 2*np.exp(-sigma*w0*dt)*np.cos(wd*dt)
	b2 = -np.exp(-2*sigma*w0*dt)
	S0 = np.exp(-sigma*w0*dt)*np.sin(wd*dt)/(wd*dt)
	if outtype == 'DIS':
		val_d_i = b1*outlist[i-1] + b2*outlist[i-2] - S0*(dt**2)*tr_bot.data[i-1]
		outlist.append(val_d_i)
	elif outtype == 'VEL':
		# aj12 = (tr_bot.data[i]+tr_bot.data[i-1])/2
		# aj32 = (tr_bot.data[i-1]+tr_bot.data[i-2])/2
		val_v_i = b1*outlist[i-1] + b2*outlist[i-2] - S0*dt*(tr_bot.data[i-1]-tr_bot.data[i-2])#(aj12-aj32)
		outlist.append(val_v_i)
	elif outtype == 'ACC':
		val_a_i = b1*outlist[i-1] + b2*outlist[i-2] - S0*(tr_bot.data[i]-2*tr_bot.data[i-1]+tr_bot.data[i-2])
		outlist.append(val_a_i)
	return outlist

def lee1990(tr_bot,w0,sigma,dt,outlist,outtype='DIS'):
	alpha = w0*sigma
	beta = w0*np.sqrt(1-sigma**2)
	T = dt
	# Z-transform parameters
	b = 2*np.exp(-alpha*T)*np.cos(beta*T)
	c = np.exp(-2*alpha*T)
	a0 = np.exp(-alpha*T)
	if outtype == 'DIS':
		gamma0 = 1
		val_d_i = b*outlist[i-1] - c*outlist[i-2] - gamma0*a0*T*tr_bot.data[i-1]
		outlist.append(val_d_i)
	elif outtype == 'VEL':
		gamma1 = 1
		val_v_i = b*outlist[i-1] - c*outlist[i-2] - gamma1*a0*(tr_bot.data[i-1]-tr_bot.data[i-2])#(aj12-aj32)
		outlist.append(val_v_i)
	elif outtype == 'ACC':
		gamma2 = 1
		val_a_i = b*outlist[i-1] - c*outlist[i-2] - gamma2*a0*(tr_bot.data[i]-2*tr_bot.data[i-1]+tr_bot.data[i-2])
		outlist.append(val_a_i)
	return outlist
	
def plot_res(dom,chan,top_obs,top_pred,drift_obs,drift_pred,evid,id,sigma,angle,xlim=[None,None]):
	if dom == 'ACC':
		outlabel = 'acceleration'
		ftag = 'acc'
	elif dom == 'VEL':
		outlabel = 'velocity'
		ftag = 'vel'
	elif dom == 'DIS':
		outlabel = 'displacement'
		ftag = 'disp'

	fig, axs = plt.subplots(2,1,sharex=True,sharey=False,dpi=300,figsize=(12,6))
	axs[0].plot(top_obs,color='blue',label=f'{chan} {outlabel}')#tr_bot.times(),
	axs[0].plot(top_pred,color='orange',label=f'{chan} {outlabel} simulated')
	axs[1].plot(drift_obs,color='blue',label=f'{chan} drift')
	axs[1].plot(drift_pred,color='orange',label=f'{chan} drift simulated')
	axs[0].legend(loc='upper right')
	axs[1].legend(loc='upper right')
	axs[0].set_title(f'Evid:{evid} | ID:{id} | W0:{T0s[chan[-1]]} | Sigma:{sigma} | Angle:{angle}')
	if xlim != [None,None]:
		axs[0].set_xlim(xlim)
	plt.tight_layout()
	plt.savefig(f'../Figures/Relative_Displacement/{evid}/{id}_{ftag}.png')
	plt.close(fig)
	return

freqmin = 1
freqmax = 15

# Load Event Catalog
ev_db = pd.read_csv('../DBs/EQCatalog/catalog2021-2024.csv', skiprows=range(1, 2131))
ev_db[['Time']] = ev_db[['Time']].apply(pd.to_datetime)

# Sentinella Building Info
building_db = pd.read_csv('../DBs/sentinella_db.csv',sep=',')
building_db['Angle'].fillna(building_db['GearthAngle'], inplace=True)

# Reverse Channels
reversed_chan = {'CORD':'X','MNFL':'Y','PRPN':'X','UNIU':'N'}#,'UNIU':'E'
path = '../DBs/EQData/'

# Read an earthquake
years = ['2024']
evids = ['2130']
stas = ['TOLM']
# evids = []; years = []
# for folder in glob.glob('../DBs/EQData/*/*'):
# 	evids.append(folder.split('/')[-1])
# 	years.append(folder.split('/')[-2])

for year, evid in tqdm(zip(years,evids)):
	# Emtpy DataFrame 4 results
	db_ev = pd.DataFrame(columns=['Station','Channel','PGABot','PGATop','PGATopReal','PGARat','PGARatReal','PGDBot','PGDTop','PGDTopReal','PGDRat','PGDRatReal'])
	stanames = []; channelnames = []; 
	pgabots = []; pgatops = []; pgatopreals = []; pgaratpred = []; pgaratreal = []
	pgdbots = []; pgdtops = []; pgdtopreals = []; pgdratpred = []; pgdratreal = []

	# mseeds = os.listdir(os.path.join(path,year,evid))
	# stas = []
	# for mseed in mseeds:
	# 	sta = mseed.split('.')[0]
	# 	if sta not in stas:
	# 		stas.append(sta)
	for sta in stas:
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
			f0x = sta_build_db['f0 building X [hz]'].iloc[0]
			f0y = sta_build_db['f0 building Y [hz]'].iloc[0]
			angle = sta_build_db['Angle'].iloc[0]
			f0sx = [float(i) for i in f0x.split('-')]
			f0sy = [float(i) for i in f0y.split('-')]
			T0x = round(((f0sx[1]-f0sx[0])/2)+f0sx[0],2)
			T0y = round(((f0sy[1]-f0sy[0])/2)+f0sy[0],2)
			sigmax = 0.03 # For TOLM
			sigmay = 0.05 # For TOLM

			w0s = {'X':T2w(T0x),'Y':T2w(T0y)}
			T0s = {'X':T0x,'Y':T0y}
			# print(sta,n_floor,f0sx,T0x,f0sy,T0y,angle)
			
			# Get Top-Bottom Waveform
			st_bottom = st.select(location=locs[0])
			st_top = st.select(location=locs[1])
			# Reverse Channels
			for tr in st_top:
				if tr.stats.station in list(reversed_chan.keys()):
					if tr.stats.channel[-1] == reversed_chan[tr.stats.station]:
						tr.data = -tr.data

			anguse = GetBackazimuth(angle)
			if np.isnan(angle):
				x,y = rotate_signals(st_bottom[1].data, st_bottom[0].data, 0)
				st_bottom[0].data = y
				st_bottom[1].data = x
			else:
				x,y = rotate_signals(st_bottom[1].data, st_bottom[0].data, anguse)
				st_bottom[0].data = y
				st_bottom[1].data = x
			

			st_bottom, st_bottom_vel, st_bottom_disp = acc2disp(st_bottom,freqmin=freqmin,freqmax=freqmax)
			st_top, st_top_vel, st_top_disp = acc2disp(st_top,freqmin=freqmin,freqmax=freqmax)			

			for tr_bot,tr_bot_vel,tr_bot_disp,tr_top,tr_top_vel,tr_top_disp in zip(st_bottom,st_bottom_vel, st_bottom_disp,st_top,st_top_vel,st_top_disp):
				top_pred_disp = [0,0]
				top_pred_vel = [0,0]
				top_pred_acc = [0,0]
				if tr_top.stats.channel[-1] == 'X':
					sigma = sigmax
				elif tr_top.stats.channel[-1] == 'Y':
					sigma = sigmay
				dt = tr_bot.stats.delta
				w0 = w0s[tr_top.stats.channel[-1]]
				for i in range(tr_bot.stats.npts):
					if i > 1:
						top_pred_dis = jin2004(tr_bot,w0,sigma,dt,top_pred_dis,outtype='DIS')
						top_pred_vel = jin2004(tr_bot,w0,sigma,dt,top_pred_vel,outtype='VEL')
						top_pred_acc = jin2004(tr_bot,w0,sigma,dt,top_pred_acc,outtype='ACC')
						# top_pred_dis = lee1990(tr_bot,w0,sigma,dt,top_pred_dis,outtype='DIS')
						# top_pred_vel = lee1990(tr_bot,w0,sigma,dt,top_pred_vel,outtype='VEL')
						# top_pred_acc = lee1990(tr_bot,w0,sigma,dt,top_pred_acc,outtype='ACC')
				
				tr_top_pred_disp = Trace(data=np.array(top_pred_disp))
				tr_top_pred_disp.stats.delta = dt
				tr_top_pred_disp.filter('bandpass',freqmin=freqmin,freqmax=freqmax)
				tr_top_pred_disp.detrend('linear')
				tr_top_pred_disp.taper(0.05)

				tr_top_pred_vel = Trace(data=np.array(top_pred_vel)) # LOOK AT THE NEGATIVE SIGN!
				tr_top_pred_vel.stats.delta = dt
				tr_top_pred_vel.filter('bandpass',freqmin=freqmin,freqmax=freqmax)
				tr_top_pred_vel.detrend('linear')
				tr_top_pred_vel.taper(0.05)
				
				tr_top_pred_acc = Trace(data=np.array(top_pred_acc)) # LOOK AT THE NEGATIVE SIGN!
				tr_top_pred_acc.stats.delta = dt
				tr_top_pred_acc.filter('bandpass',freqmin=freqmin,freqmax=freqmax)
				tr_top_pred_acc.detrend('linear')
				tr_top_pred_acc.taper(0.05)

				stanames.append(sta)
				channelnames.append(tr_top.stats.channel[-1])
				pgd_bot = max(abs(tr_bot_disp.data))
				pgd_top = max(abs(tr_top_pred_disp.data + tr_bot_disp.data))
				pgd_top_real = max(abs(tr_top_disp.data))
				pgdbots.append(pgd_bot)
				pgdtops.append(pgd_top)
				pgdtopreals.append(pgd_top_real)
				pgdratreal.append(pgd_top_real/pgd_bot)
				pgdratpred.append(pgd_top/pgd_bot)
				pga_bot = max(abs(tr_bot.data))
				pga_top = max(abs(tr_top_pred_acc.data + tr_bot.data))
				pga_top_real = max(abs(tr_top.data))
				pgabots.append(pga_bot)
				pgatops.append(pga_top)
				pgatopreals.append(pga_top_real)
				pgaratreal.append(pga_top_real/pga_bot)
				pgaratpred.append(pga_top/pga_bot)

				# DISPLACEMENT
				disp_meas = tr_top_pred_disp.data + tr_bot_disp.data 
				drift_calc = tr_top_pred_disp.data# - tr_bot_disp.data
				drift_meas = tr_top_disp.data - tr_bot_disp.data
				# ACCELERATION
				acc_meas = tr_top_pred_acc.data + tr_bot.data
				acc_drift_calc = tr_top_pred_acc.data# - tr_bot_disp.data
				acc_drift_meas = tr_top.data - tr_bot.data
				# VELOCITY
				vel_meas = tr_top_pred_vel.data + tr_bot_vel.data
				vel_drift_calc = tr_top_pred_vel.data# - tr_bot_disp.data
				vel_drift_meas = tr_top_vel.data - tr_bot_vel.data

				plot_res('DIS',tr_top_disp.stats.channel,tr_top_disp.data,disp_meas,drift_meas,drift_calc,evid,tr_top.id,sigma,angle,xlim=[0,3000])
				plot_res('VEL',tr_top_disp.stats.channel,tr_top_vel.data,vel_meas,vel_drift_meas,vel_drift_calc,evid,tr_top.id,sigma,angle,xlim=[0,3000])
				plot_res('ACC',tr_top_disp.stats.channel,tr_top.data,acc_meas,acc_drift_meas,acc_drift_calc,evid,tr_top.id,sigma,angle,xlim=[0,3000])
	# Save Results in a CSV file
	db_ev['Station'] = stanames
	db_ev['Channel'] = channelnames
	db_ev['PGABot'] = pgabots
	db_ev['PGATop'] = pgatops
	db_ev['PGATopReal'] = pgatopreals
	db_ev['PGARat'] = pgaratpred
	db_ev['PGARatReal'] = pgaratreal
	db_ev['PGDBot'] = pgdbots
	db_ev['PGDTop'] = pgdtops
	db_ev['PGDTopReal'] = pgdtopreals
	db_ev['PGDRat'] = pgdratpred
	db_ev['PGDRatReal'] = pgdratreal
	if len(db_ev) > 0:
		db_ev.to_csv(f'../DBs/GMP_Ratios/{evid}.csv',index=False)
