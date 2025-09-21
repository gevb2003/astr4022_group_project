# Import required packages (same as those in eos.py)
import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as c
import astropy.units as u
import scipy.optimize as op
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import astropy.io.fits as pyfits
from astropy.table import Table

# Import functions defined in eos.py and opac.py
from eos import *
from opac import *

#-----------------------STELLAR PARAMETERS INPUT-----------------------------
Teff = 3000 # in units of kelvin
logg = 1.0

# toggle for plotting density, pressure and tempeature profiles
plot_profiles = False

#-----------------------END OF INPUT-----------------------------

# derived parameters
g_cgs = 10**logg * u.cm/u.s**2


# import functions to read opacity tables from Grace
from opacity_reader import *
# Read in solar metallicity file 
log_T, log_R, opac_table = read_opacity_table('caffau11.7.02.tron')
# Define ranges of T and R 
Tmin, Tmax = 10**log_T.min(), 10**log_T.max()
Rmin, Rmax = 10**log_R.min(), 10**log_R.max()


# Set to solar abundance 
mu_sol = 1.3

# Define function to get rosseland mean opacity from a pressure and optical depth
def kappa_from_P_tau(P, tau):
	"""
	Parameters
	----------
		P : Pressure in cgs units
		tau : Optical depth scalar

	Returns
	---------
		kappa : Rosseland mean opacity in cgs units interpolated from tables by Caffau et al (2011)

	"""


	# Calculate the tempeature assuming a grey atmosphere
	T = Teff * (3/4 * (tau + 2/3))**(1/4)

	# Calculate density assuming the ideal gas law
	rho_cgs = (P*u.dyne/u.cm**2 * mu_sol * c.m_p/(c.k_B * T*u.K)).cgs

	T6 = T/1e6
	R = rho_cgs.value / (T6**3)

	# Check that given T and R values are within the range of the opacity table
	if (T < Tmin) or (T > Tmax): 
		raise ValueError(f'Temperature associated with given pressure and optical depth is outside the opacity table range. Please choose another pair of inputs')

	if (R < Rmin) or (R > Rmax): 
		raise ValueError(f'R value associated with given pressure and optical depth is outside the opacity table range. Please choose another pair of inputs')

	# Find rosseland mean opacity for the corresponding temperaure and density 
	log_kappa = interp_opac_R(T, R, log_T, log_R, opac_table) 
	kappa = 10**log_kappa 

	return kappa


# Define function to get the derivative dPdtau = g/kappa
def dpdtau(P, tau):
	
	# Get opacity from given parameters
	kappa = kappa_from_P_tau(P, tau)
	# define derivative in cgs units
	derivative = g_cgs.value/kappa

	return derivative

# estimate the optical depth of the smallest pressure in the opacity tables
P_init = 25 # cgs 
frac_tol = 0.02 # fractional tolerance of convergence 

small_tau = np.linspace(1e-4, 1e-2, 100)

for tau in small_tau:

	try:
		kappa = kappa_from_P_tau(P_init, tau)
	except ValueError as e:
		print(e)
		continue  # skip this tau if it's outside the table

	# Constant kappa over optical depth step with 
	P = g_cgs.value * tau / kappa

	# retain optical depth value if it agrees with the initial pressure within the chosen tolerance
	if np.abs(P - P_init) <= frac_tol * P_init:
		tau_init = tau
		break 
	else:
		continue

# Initiate tau grid from this starting point
tau_grid = np.linspace(tau_init, 5, 100)

# solve differential equation with initial condition. 
soln = solve_ivp(dpdtau, [tau_grid[0],tau_grid[-1]], [P_init], t_eval=tau_grid)

# extract solution and optical depth grid
p = soln.y[0]*u.dyne/u.cm**2
tau = soln.t

# Grey temperture profile
Temp = Teff * (3/4 * (tau + 2/3))**(1/4)

# compute density assuming the ideal gas law 
rho = p * mu_sol * c.m_p/(c.k_B * Temp*u.K)





#----------------PLOTS-------------------

if plot_profiles == True:

	# set up subplots
	fig, ax = plt.subplots(1,3,figsize=(15,5))

	# pressure
	ax[0].plot(tau_grid, p.cgs.value)
	ax[0].set_xlabel('Optical depth')
	ax[0].set_ylabel('Pressue (dyne/cm2)')

	# density
	ax[1].plot(tau_grid, rho.cgs.value)
	ax[1].set_xlabel('Optical Depth')
	ax[1].set_ylabel('Density (g/cm3)')

	# temperature
	ax[2].plot(tau_grid, Temp)
	ax[2].set_xlabel('Optical Depth')
	ax[2].set_ylabel('Temperature (K)')

	plt.tight_layout()
	fig.savefig("atmosphere_profiles.png", dpi=300, bbox_inches="tight")


#----------------------EQUATION OF STATE---------------------


# get pressure arrays for atomic and molecular species also a function of optical depth

# key for pressure output
key = ['e', 'H', 'He', 'C', 'N', 'O', 'Ne', 'Na', 'Mg', 'Si', 'S', 'K', 'Ca', 'Fe', 'Ti', 
	   'H+', 'He+', 'C+', 'N+', 'O+', 'Ne+', 'Na+', 'Mg+', 'Si+', 'S+', 'K+', 'Ca+', 'Fe+', 'Ti+',
	   'NN', 'TiO', 'TiO2', 'MgN', 'CaH', 'HH', 'CO', 'HOH', 'OH', 'H-']

# initiate 2d table to store number densities 
ns = np.empty(shape=(len(key), len(tau_grid)))

for i in range(len(tau_grid)):
	# get log pressure in cgs units of the various atomic and molecular species 
	logp = equilibrium_solve(rho[i], Temp[i]*u.K, plot=False)
	p = 10**logp*u.dyne*u.cm**-2

	# convert pressures to number densities assuming ideal gas law
	n = (p/(c.k_B * Temp[i]*u.K)).cgs

	# append entire row to table
	ns[:, i] = n.value
