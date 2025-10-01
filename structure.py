# Let's compute the structure of an atmosphere, using the folowing modules and assumptions
# 1) A grey atmosphere and hydrostatic equilibrium (analytic)
# 2) An equation of state using the Saha equation: tabulated in rho_Ui_mu_ns_ne.fits
# 3) Opacities computed using the methods in opac.py: tabulated in Ross_Planck_opac.fits
#
# For speed, units are:
# - Length: cm
# - Mass: g
# - Time: s
# - Temperature: K
# - Frequency: Hz

from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import solve_ivp, cumulative_trapezoid
import astropy.units as u
import astropy.constants as c
import numpy as np
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import opac
from scipy.special import expn
from strontium_barium import *
plt.ion()
from opacity_reader import *
from astropy.io import fits
import eos as eos

# === USER INPUT ===
Teff = 5000 # in units of kelvin # Mike says: 4850 is the lowest. 9000 is the highest that makes sense.
logg = 1.0
g = 10**logg * u.cm/u.s**2
P0 = 10 # Initial pressure in dyn/cm^2
convective_cutoff = 1.3 # Limits T due to onset of convection. If 2.0 then there is no cutoff.
# === END OF INPUT ===



# === Load opacities ===
T_grid_chi, R_grid_chi, kappa_bar_l = read_opacity_table('rosseland_opacities/Caffau11/caffau11.7.02.tron')
# RectBivariateSpline requires ascending order. Need to flip T_grid_chi and chi_bar_l.
T_grid_chi = T_grid_chi[::-1]
kappa_bar_l = kappa_bar_l[::-1,:] # previously named kappa_bar_ross

# === Construct the log(P) and T vectors ===
# f_opac = pyfits.open('Ross_Planck_opac.fits')
#h = f_opac[0].header
#T_grid = h['CRVAL1'] + np.arange(h['NAXIS1'])*h['CDELT1']
#Ps_log10 = h['CRVAL2'] + np.arange(h['NAXIS2'])*h['CDELT2']
# === OLD VERSION ===

T_grid = T_grid_chi # log K
R_grid = R_grid_chi # log (rho/T6^3)

#Create our interpolator functions
kappa_bar_l_interp = RectBivariateSpline(T_grid_chi, R_grid_chi, kappa_bar_l)
#f_kappa_bar_Ross = RegularGridInterpolator((Ps_log10, T_grid), kappa_bar_Ross)

def T_tau(tau, Teff):
	"""
	Temperature for a simplified grey atmosphere, with an analytic
    approximation for the Hopf q (feel free to check this!)
	"""
	q = 0.71044 - 0.1*np.exp(-2.0*tau)
	T = (0.75*Teff**4*(tau + q))**.25
	return T

def mu_from_P_T(P, T):
    """
    Given a pressure (in CGS units, value only) and T, return mu from the eos table.
    For the purpose of solve_ivp, so needs to handle lists/arrays.
    """
    if not isinstance(P, u.Quantity):
        P = [p_cgs * u.dyne / u.cm**2 for p_cgs in P] # Ensure it has units for ns_from_P_T

    mu = [] # Create empty mu list
    for pressure in P:
        n_e, n_s, mu_val, Ui, rho = eos.ns_from_P_T(pressure, T) # update this to a better function to use
        mu.append(mu_val)
    return mu

def dPdtau_OLD(_, P, T):
	"""
	Compute the derivative of pressure with respect to optical depth.
	"""
	kappa_bar = f_kappa_bar_Ross((P, T))
	return g / kappa_bar

def get_R(P, T):
	"""
	Convert pressure (in CGS units, value only) and T to log R.
	For the purpose of solve_ivp, so needs to handle lists/arrays.
	"""
	if not isinstance(P, u.Quantity):
		P = [p_cgs * u.dyne / u.cm**2 for p_cgs in P] # Ensure it has units for ns_from_P_T

	R = [] # Create empty R list
	for pressure in P:
		n_e, n_s, mu, Ui, rho = eos.ns_from_P_T(pressure, T) # update this to a better function to use
		R.append(np.log10((rho/(T.to(u.MK)**3)).value))
	return R

def dPdtau(_, P, T):
    """
	Compute the derivative of pressure with respect to optical depth.
	"""
    R = get_R(P, T)

    kappa = []
    for rho in R:
          kappa_bar = kappa_bar_l_interp(np.log10(T.to(u.K).value), R, grid=False)
          kappa.append(kappa_bar)
    return g / kappa

def get_min_P(R, T):
    """
    Given a log R and T, return the minimum pressure in the opacity table.
    """
    rho = 10**R * (T.to(u.MK)**3).value
    
    return P # UNFINISHED
P0 = np.maximum(10**(Ps_log10[0]), P0)  # Ensure P0 is not less than the minimum pressure in the table
# NEED TO DEFINE A FUNCTION TO FIND THE ABOVE

# Starting from the lowest value of log(P), integrate P using solve_ivp
#solve_ivp(fun, t_span, y0, method='RK45', t_eval=None, dense_output=False, events=None, vectorized=False, args=None, **options)
tau_grid = np.concatenate((np.arange(3)/3*1e-3,np.logspace(-3,1.3,30)))
sol = solve_ivp(dPdtau, [0, 20], [P0], args=(Teff,), t_eval=tau_grid, method='RK45')
Ps = sol.y[0]
Ts = T_tau(tau_grid, Teff)
# Artificially cut the deep layer temperature due to convection.
Ts = np.minimum(Ts,convective_cutoff*Teff)

# Load the equation of state
f_eos = pyfits.open('rho_Ui_mu_ns_ne.fits')
rho = f_eos['rho [g/cm**3]'].data

# Add interpolation functions for whatever isn't already in opac - just rho.
f_rho = RegularGridInterpolator((Ps_log10, T_grid), rho)

# Interpolate onto the tau grid
kappa_bars = f_kappa_bar_Ross((np.log10(Ps), Ts))
rhos = f_rho((np.log10(Ps), Ts))    

# First, lets plot a continuum spectrum
wave = np.linspace(50, 2000, 1000) * u.nm  # Wavelength in nm
flux = np.zeros_like(wave)  # Initialize flux array

# Just like in grey_flux.py, but in frequency 
planck_C1 = (2*c.h*c.c**2/(1*u.um)**5).si.value
planck_C2 = (c.h*c.c/(1*u.um)/c.k_B/(1*u.K)).si.value

# Planck function, like in grey_flux.py
def Blambda_SI(wave_um, T):
    """
    Planck function in cgs units.
    """
    return planck_C1/wave_um**5/(np.exp(planck_C2/wave_um/T)-1)

def compute_H(wave, Ts, tau_grid, kappa_nu_bars, kappa_bars):
    Hlambda = np.zeros(len(wave))  # Initialize H array
    # Compute the flux for each wavelength
    for i, w in enumerate(wave):
        # Now we need S(tau_nu), i.e. B(tau_nu(tau))
        tau_nu  = cumulative_trapezoid(kappa_nu_bars[i]/kappa_bars, x=tau_grid, initial=0)
        wave_um = w.to(u.um).value
        Slambda = Blambda_SI(wave_um, Ts)
        Hlambda[i] = 0.5*(Slambda[0]*expn(3,0) + \
		np.sum((Slambda[1:]-Slambda[:-1])/(tau_nu[1:]-tau_nu[:-1])*\
			(expn(4,tau_nu[:-1])-expn(4,tau_nu[1:]))))
    return Hlambda

print("Computing continuum opacities")
kappa_nu_bars = np.empty((len(wave), len(tau_grid)))
for i, (T, P, rho) in enumerate(zip(Ts, Ps, rhos)):
    kappa_nu_bars[:,i] = opac.kappa_cont((c.c/wave).to(u.Hz).value, np.log10(P), T)/rho
print("Computing continuum spectrum")
H = compute_H(wave, Ts, tau_grid, kappa_nu_bars, kappa_bars)

# Plot the flux and the blackbody approximation
# So far it isn't great... why?
plt.figure(1)
plt.clf()
plt.plot(wave, 4*np.pi*H /1e6, label='Flux')
plt.plot(wave, np.pi*Blambda_SI(wave.to(u.um).value, Teff) / 1e6, label='Blackbody')
plt.xlabel('Wavelength (nm)')
plt.ylabel(r'Flux (W/m$^2$/$\mu$m)')
plt.legend()
plt.show()

# Now lets add in all lines. The Strontium line calculation is saved under "week34" if you
# want to look at that.
nu0 = 3e8/700e-9
dlnu = 5e-6
Nnu = 200000
print("Computing line opacities")

# Compute the frequency grid
nu = nu0 * np.exp(dlnu * np.arange(Nnu))
wave_nm= (c.c/(nu*u.Hz)).to(u.nm).value
kappa_all_nu_bars = np.empty((Nnu, len(tau_grid)))
for i, (T, P, rho) in enumerate(zip(Ts, Ps, rhos)):
    kappa_all_nu_bars[:,i] = opac.kappa_cont(nu, np.log10(P), T) + \
        opac.weak_line_kappa(nu0, dlnu, Nnu, np.log10(P), T) + \
        opac.strong_line_kappa(nu0, dlnu, Nnu, np.log10(P), T)
    kappa_all_nu_bars[:,i] /= rho

print("Computing Line Spectrum")
H_all = compute_H(wave_nm*u.nm, Ts, tau_grid, kappa_all_nu_bars, kappa_bars)

# Add in a macroturbulence and rotation convolution. 
# 43 is WiFeS. 2.0 is a perfect spectrograph.
macroturb = 43.0 
macroturb = 2.0
width = macroturb / 3e5 / dlnu
g_macro = np.exp(-(np.arange(-int(2.5*width), int(2.5*width)+1)**2)/width**2)
H_all = np.convolve(H_all, g_macro/np.sum(g_macro), mode='same')

#Plot this
plt.figure(2)
plt.clf()
plt.plot(wave_nm, 4*np.pi*H_all / 1e6, label='Flux')
plt.xlabel('Wavelength (nm)')
plt.ylabel(r'Flux (W/m$^2$/$\mu$m)')
plt.legend()
plt.show()