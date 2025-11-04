from obspy import read, Trace, Stream
import numpy as np

def acc2disp(st,freqmin=0.8,freqmax=10):
	st.detrend('linear')
	st.taper(max_percentage=0.05)
	st.filter('bandpass',freqmin=freqmin,freqmax=freqmax)
	st.taper(max_percentage=0.05)
	st_vel = st.copy()
	# Acceleration to Velocity
	st_vel.integrate('cumtrapz')
	st_disp = st_vel.copy()
	# Velocity to Displacement
	st_disp.integrate('cumtrapz')
	st_disp.detrend('linear')
	return st, st_vel, st_disp

def f0h2vs(f,h):
	return 4*f*h

def T2w(T):
	return 2*np.pi/T

def f2w(f):
	return 2*np.pi*f

def w2f(w):
	return w/(2*np.pi)

def w2T(w):
	return (2*np.pi)/w

def H2F0(H,N_LAYER):
	tot_h = H*N_LAYER
	return 1/((tot_h/3)*0.1)

def Q2zeta(Q):
	return 1/(2*Q)

def zeta2Q(zeta):
	return 1/(2*zeta)

def calc_fft(data,npts,sr):
	'''
	data: time series signal
	npts: number of points in time series
	sr: sampling rate
	'''
	Fdat = np.fft.fft(data,npts)
	freq = np.fft.fftfreq(npts, d=1./sr)
	return Fdat, freq

def calc_rfft(data,npts,sr):
	'''
	data: time series signal
	npts: number of points in time series
	sr: sampling rate
	'''
	Fdat = np.fft.rfft(data,npts)
	freq = np.fft.rfftfreq(npts, d=1./sr)
	return Fdat, freq

def calc_absfft(data,npts,sr):
	Fdat = np.fft.rfft(data)
	freq = np.fft.rfftfreq(npts, d=1./sr)
	return 2.0/sr * np.abs(Fdat), freq

def calc_ifft(data):
	'''
	data: FFT of a signal
	'''
	return np.fft.ifft(data) #irfft

def calc_irfft(data):
	'''
	data: FFT of a signal
	'''
	return np.real(np.fft.ifft(data))
