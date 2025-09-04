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
Teff = 3000 * u.K
logg = 1.0

#-----------------------END OF INPUT-----------------------------

# derived parameters
g_cgs = 10**logg * u.cm/u.s**2


# define tau grid
tau_grid = np.linspace(0,5,100)
#define grid for computing
prl_grid = -10 * np.ones(len(tau_grid))

# grey temperature profile 
T_grid = Teff * (3/4 * (tau_grid + 2/3))**(1/4)

# Solve for density as a function of optical depth 
# use solve_ivp like Mike's code
# estimate initial guess from hydrostatic equilibrium ???
# boundary conditions (surface of star rho(tau=0) = 0)


# Table required to import for finding density and pressure as functions of optical depth 
# rosseland mean opacity
# mu (average atomic weight)
# as functions of temperature and pressure (?) 

# for the purposes of starting the code use the same as Mike

#Load in the tables
archive = np.load('chi_mu_T_prl.npz')
chi_bar_l = archive['arr_0']
mu = archive['arr_1']
T_grid = archive['arr_2']
prl = archive['arr_3']
chi_bar_l_interp = RectBivariateSpline(prl, T_grid, chi_bar_l)
mu_interp = RectBivariateSpline(prl, T_grid, mu)


#A function to find the Rosseland mean chi_bar
def get_chi_bar(T, p_cgs):
	"""
	Convert pressure (in CGS units, value only) to log base 10, and interpolate.
	"""
	prl = np.log10(p_cgs)
	chi_bar = 10**(chi_bar_l_interp(prl, T.to(u.K).value, grid=False))*u.cm**2/u.g
	return chi_bar

#A function to find the tau derivative.
def dpdtau(tau, p_cgs):
	"""
	Find dpdtau, assuming global variables g and Teff
	"""
	T = Teff * (3/4 * (tau + 2/3))**(1/4)
	chi_bar = get_chi_bar(T, p_cgs)
	return [(g_cgs/chi_bar).to(u.dyne/u.cm**2).value]


#Use solve_ivp (better than Euler's method) to solve for p(tau) and state variables
#Start at a "very low" pressure as an initial guess
soln = solve_ivp(dpdtau, [0,tau_grid[-1]], [1e-5])
p = soln.y[0]*u.dyne/u.cm**2
tau = soln.t
T = Teff * (3/4 * (tau + 2/3))**(1/4) 
N = (p/c.k_B/T).cgs
rho = (N*u.u*mu_interp(np.log10(p.value), T.value, grid=False)).cgs
chi_bar_R = get_chi_bar(T, p.cgs.value)

