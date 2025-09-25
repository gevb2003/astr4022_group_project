"""
In this script, we will integrate the grey atmosphere structure.

dp/dz = -g rho
dp/dtau = g rho / chi
	= g / chi_bar
	
Then, we will add in the Ca II K line and see what the spectrum 
(and specific intensity) looks like.
"""

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as c
import eos as eos
from scipy.interpolate import RectBivariateSpline
from scipy.integrate import solve_ivp
from scipy.special import voigt_profile
from scipy.integrate import cumulative_trapezoid
from scipy.special import expn
from opacity_reader import *

#From earlier in the course
import saha_eos as saha
# Might want to update the above to use regular EOS with the chemical equilibrium solver

#Our stellar parameters
logg = 1.7
Teff = 6500 * u.K

#An optical depth grid
tau_grid = np.arange(40)/10
prl_grid = -10 * np.ones(len(tau_grid))

#Derived parameters
g = 10**logg * u.cm/u.s**2




# ===== WORK IN PROGRESS =====
# Load in the tables
# For now, we need to use the old interpolator file for mu
archive = np.load('chi_mu_T_prl.npz')
mu = archive['arr_1']
T_grid_mu = archive['arr_2']
prl_mu = archive['arr_3']
mu_interp = RectBivariateSpline(prl_mu, T_grid_mu, mu)

# Then import our own Rosseland Mean Opacity table
T_grid_chi, R_grid_chi, chi_bar_l = read_opacity_table('rosseland_opacities/Caffau11/caffau11.7.02.tron')
# RectBivariateSpline requires ascending order. Need to flip T_grid_chi and chi_bar_l.
T_grid_chi = T_grid_chi[::-1]
chi_bar_l = chi_bar_l[::-1,:]
chi_bar_l_interp = RectBivariateSpline(T_grid_chi, R_grid_chi, chi_bar_l)

#rho_grid_chi = R_to_rho(R_grid_chi, T_grid_chi)
# Get pressure and other values from eos functions
#P, n_e, ns, mu, Ui, gamma1, gamma2, gamma3, gamma1_old = eos.eos_rho_T(rho_grid_chi, T_grid_chi, full_output=True)

#A function to find the Rosseland mean chi_bar
def get_chi_bar(T, p_cgs):
	"""
	Convert pressure (in CGS units, value only) to R in log base 10, and interpolate.
	Since our new interpolator is on an R grid, we need to convert P to R.
	"""
	# Check if p has units (required for ns_from_P_T)
	if not isinstance(p_cgs, u.Quantity):
		p_cgs = p_cgs * u.dyne / u.cm**2
	n_e, n_s, mu, Ui, rho = eos.ns_from_P_T(p_cgs, T)
	R = np.log10((rho/(T.to(u.MK)**3)).value)
	chi_bar = 10**(chi_bar_l_interp(R, np.log10(T.to(u.K).value), grid=False))*u.cm**2/u.g
	return chi_bar
# ============================

#A function to find the tau derivative.
def dpdtau(tau, p_cgs):
	"""
	Find dpdtau, assuming global variables g and Teff
	"""
	T = Teff * (3/4 * (tau + 2/3))**(1/4)
	chi_bar = get_chi_bar(T, p_cgs)
	return [(g/chi_bar).to(u.dyne/u.cm**2).value]


#Use solve_ivp (better than Euler's method) to solve for p(tau) and state variables
#Start at a "very low" pressure.
soln = solve_ivp(dpdtau, [0,10], [1e-5])
p = soln.y[0]*u.dyne/u.cm**2
tau = soln.t
T = Teff * (3/4 * (tau + 2/3))**(1/4) 
N = (p/c.k_B/T).cgs
rho = (N*u.u*mu_interp(np.log10(p.value), T.value, grid=False)).cgs
chi_bar_R = get_chi_bar(T, p.cgs.value)

#Make a plot and discuss!
plt.figure(1)
plt.clf()
plt.loglog(tau[1:], N[1:])
plt.xlabel(r'$\tau_R$')
plt.ylabel(r'N (cm$^{-3}$)')
plt.title('Canopus-like grey atmosphere')
plt.tight_layout()

#------------ From Lecture 7 ---------------
#Now lets make a frequency grid, based on a wavelength grid.
nwave = 1000
wave = np.linspace(392,395,nwave)*u.nm
nu = (c.c/wave).cgs

#Central frequency of our transition
nu0 = (c.c/(3933*u.AA)).cgs

#Gamma from NIST
Gamma = 1.57e8*u.Hz

#Doppler \Delta \nu
Dnu_Doppler = (np.sqrt(2*c.k_B*(6500*u.K)/(40*u.u))/c.c*nu0).cgs

#Lets compute the profile!
phi = voigt_profile((nu-nu0).to(u.Hz).value, (Dnu_Doppler/np.sqrt(2)).to(u.Hz).value, Gamma.to(u.Hz).value/4/np.pi)
phi = phi*u.s

#From Malika and others - to check!
Blu = 4.5e10*u.s/u.g
cross_sect = (Blu * c.h*nu/4/np.pi*phi).cgs
#----------------------------------------------
#To convert the cross-section to a mass-weighted, opacity, we'll need the number density 
#of CaII as a function of \tau
abund, masses, n_p, ionI, ionII, gI, gII, gIII, elt_names = saha.solarmet()

#Find the number ensity index of Ca II. Except for hydrogen, 
#Neutrals are 3*ix-1, first ionized are 3*ix
Ca_ix = np.where(elt_names=='Ca')[0][0]
CaII_ix = 3*Ca_ix 

#Now create and fill the number density array. Here we could also do a check that
#our Saha equation mu matches the tabulated one!
N_CaII = np.empty_like(N)
for i, (Ti, rhoi) in enumerate(zip(T, rho)):
	n_e, ns, mu, Ui = saha.ns_from_rho_T(rhoi,Ti)
	N_CaII[i] = ns[CaII_ix]
	
#Check that the Ca II ionisation fraction makes sense...
plt.figure(2)
plt.clf()
plt.plot(T, N_CaII/N / np.max(N_CaII/N))
plt.ylabel('Ca single ionized fraction')
plt.xlabel('T (K)')

#Now, finally we can compute our wavelengt-dependent chi_bar! We'll approximate the
#continuum opacity as the Rosseland mean opacity
chi_bar_nu = chi_bar_R + (N_CaII/rho) * np.repeat(cross_sect, len(N)).reshape(nwave,len(N))

#Next, plot this opacity as a function of wavelength.
plt.figure(3)
plt.clf()
plt.semilogy(wave, chi_bar_nu)
plt.xlabel('Wavelength (nm)')
plt.ylabel(r'$\bar{\chi}_\nu$ (cm$^2$/g)')

#To compute a line profile, we need flux, which means we first need tau_nu
#For simplicity, lets just do a trapezoidal rule integration
tau_nu = cumulative_trapezoid((chi_bar_nu/chi_bar_R).value, tau, initial=0)

#Plot this optical depth as a function of wavelength.
plt.figure(4)
plt.clf()
plt.semilogy(wave, tau_nu)
plt.xlabel('Wavelength (nm)')
plt.ylabel(r'$\tau_\nu$ (cm$^2$/g)')
plt.axis([392,395,0.01,100])

#To do an LTE calculation we'll need the Planck function (equal to the source function)
def Bnu(nu, T):
	return (2*c.h*nu**3/c.c**2 / np.exp(c.h*nu/c.k_B/T)).cgs

#Finally, to compute flux, we'll either need a 2D integral, or the Phi transform.
#The Phi transform is a lot more computationally efficient! 
#First integrate using the trapezoidal rule, but also integrate where we
#approximate the source function as being linear between each layer, making an
#explicit integral
print('Computing Flux')
Fnu_trapz = np.empty( nwave )
Fnu = np.empty( nwave )
for i in range(nwave):
	Snu_cgs = Bnu(c.c/wave[i], T).to(u.erg/u.cm**2).value
	Fnu_trapz[i] = 2*np.pi*np.trapz(Snu_cgs*expn(2,tau_nu[i]),tau_nu[i])
	Fnu[i] = 2*np.pi*(Snu_cgs[0]*expn(3,0) + \
		np.sum((Snu_cgs[1:]-Snu_cgs[:-1])/(tau_nu[i,1:]-tau_nu[i,:-1])*\
			(expn(4,tau_nu[i,:-1])-expn(4,tau_nu[i,1:]))))
Fnu_trapz *= u.erg/u.cm**2
Fnu *= u.erg/u.cm**2
Flambda_trapz = (Fnu_trapz* c.c/wave**2).to(u.W/u.m**2/u.nm)
Flambda = (Fnu * c.c/wave**2).to(u.W/u.m**2/u.nm)
print('Flux computed')

plt.figure(5)
plt.clf()
plt.plot(wave, Flambda, label='Short Char. Explicit')
plt.plot(wave, Flambda_trapz, label='Trapezoidal')
plt.xlabel('Wavelength (nm)')
plt.ylabel(r'F$_\lambda$ (W/m$^2$/nm)')
plt.tight_layout()

