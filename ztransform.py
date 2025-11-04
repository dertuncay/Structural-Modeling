import numpy as np

def ztransform(i,trace,w0,sigma,dt,outlist,outtype='DIS'):
	"""
	Calculates drift motion for a SDOF system with a given input parameters

	Parameters:
	-----------
	i : integer
		time increment
	trace : list or np.ndarray
		signal collected at the bottom
	w0 : float
		natural frequency of SDOF oscillator
	sigma : float
		damping ratio (%)
	dt : float
		time interval
	outlist : list
		predicted motion
	outtype : string (default: 'DIS')
		domain of the predicted motion
	Returns:
	--------
	outlist : list
		predicted motion
	"""

	wd = w0*np.sqrt(1-sigma**2)
	b1 = 2*np.exp(-sigma*w0*dt)*np.cos(wd*dt)
	b2 = -np.exp(-2*sigma*w0*dt)
	S0 = np.exp(-sigma*w0*dt)*np.sin(wd*dt)/(wd*dt)
	if outtype == 'DIS':
		val_d_i = b1*outlist[i-1] + b2*outlist[i-2] - S0*(dt**2)*trace[i-1]
		outlist.append(val_d_i)
	elif outtype == 'VEL':
		val_v_i = b1*outlist[i-1] + b2*outlist[i-2] - S0*dt*(trace[i-1]-trace[i-2])
		outlist.append(val_v_i)
	elif outtype == 'ACC':
		val_a_i = b1*outlist[i-1] + b2*outlist[i-2] - S0*(trace[i]-2*trace[i-1]+trace[i-2])
		outlist.append(val_a_i)
	return outlist
