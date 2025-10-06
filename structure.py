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

from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline
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
import time
from labellines import labelLines

# === USER INPUT ===
Teff = 3300 # in units of kelvin. Minimum 3300 K
logg = 1.0
g = 10**logg * u.cm/u.s**2
P0 = 10 # Initial pressure in dyn/cm^2
convective_cutoff = 1.3 # Limits T due to onset of convection. If 2.0 then there is no cutoff.
# === END OF INPUT ===



# === Load opacities ===
T_grid_chi, R_grid_chi, kappa_bar_l = read_opacity_table('rosseland_opacities/Caffau11/caffau11.7.02.tron')
# RectBivariateSpline requires ascending order. Need to flip T_grid_chi and chi_bar_l.
#T_grid_chi = T_grid_chi[::-1]
#kappa_bar_l = kappa_bar_l[::-1,:] # previously named kappa_bar_ross

T_grid = T_grid_chi # log K - be careful with this!
R_grid = R_grid_chi # log (rho/T6^3)

#Create our interpolator functions
#kappa_bar_l_interp = RectBivariateSpline(T_grid, R_grid, kappa_bar_l) # log Kappa
# Transpose kappa to match the order of the grids
kappa_bar_l = kappa_bar_l.T
kappa_bar_l_interp = RegularGridInterpolator((R_grid, T_grid), kappa_bar_l) # log Kappa

def T_tau(tau, Teff):
	"""
	Temperature for a simplified grey atmosphere, with an analytic
    approximation for the Hopf q (feel free to check this!)
	"""
	q = 0.71044 - 0.1*np.exp(-2.0*tau)
	T = (0.75*Teff**4*(tau + q))**.25
	return T

def mu_from_P_T(P, T): 
    """ Given a pressure (in CGS units, value only) and T, return mu from the eos table. 
    For the purpose of solve_ivp, so needs to handle lists/arrays. Requires linear P and T. """ 
    P = np.atleast_1d(P) 
    T = np.atleast_1d(T) 
    mu = [] # Create empty mu list 
    for pressure in P: 
        if not isinstance(pressure, u.Quantity): 
            pressure = pressure * u.dyne/u.cm**2 
        if not isinstance(T, u.Quantity): 
            T = T * u.K 
        species, logPs, nums, mu_val = eos.P_T_equilibrium_tables(pressure, T, plot=False, verbose=False) 
        mu.append(mu_val) 
    return mu

def mu_from_P_T_pairwise(P, T):
    """
    Pairwise mu calculation: mu[i] corresponds to P[i], T[i].
    Returns a numpy array of floats.
    """
    P = np.atleast_1d(P)
    T = np.atleast_1d(T)

    if P.shape != T.shape:
        raise ValueError("P and T must have the same shape for pairwise evaluation")

    mu = np.empty(P.shape, dtype=float)
    num = []

    for i, (p_val, t_val) in enumerate(zip(P, T)):
        # Ensure scalar Quantities
        if not isinstance(p_val, u.Quantity):
            p_val = p_val * u.dyne / u.cm**2
        if not isinstance(t_val, u.Quantity):
            t_val = t_val * u.K

        # eos function expects arrays of T, but we only want one scalar here
        _, _, nums, mus = eos.P_T_equilibrium_tables(p_val, np.atleast_1d(t_val), plot=False, verbose=False)
        mu[i] = float(mus[0])  # take scalar value
        num.append(nums)

    return num, mu

def get_R(P, T):
    """
    Convert pressure (in CGS units, value only) and T to log R.
    For the purpose of solve_ivp, so needs to handle lists/arrays.
    Requires linear P and T.
    """

    R = [] # Create empty R list
    
    # Handle arrays or integers
    P = np.atleast_1d(P)
    T = np.atleast_1d(T)
    for pressure, temp in zip(P, T):
        if not isinstance(pressure, u.Quantity):
            pressure = pressure * u.dyne / u.cm**2
        if not isinstance(temp, u.Quantity):
            temp = temp*u.K
        mu = mu_from_P_T(pressure, temp)
        rho = (pressure/c.k_B/temp * u.u * mu).to(u.g/u.cm**3) # Ideal gas law
        R.append(float(np.log10((rho/(temp.to(u.MK)**3)).value)))
    R = np.array(R)

    return R

def dPdtau(_, P, T):
    """
	Compute the derivative of pressure with respect to optical depth.
    Requires linear P and T.
	"""
    R = get_R(P, T) # log R

    kappas = []
    for rho in R:
          kappa_bar = kappa_bar_l_interp((rho, np.log10(T)))
          kappa_bar = 10**kappa_bar # Convert from log kappa to kappa
          kappas.append(float(kappa_bar))
    kappas = np.array(kappas)
    return g / kappas

def get_min_P(R, T):
    """
    Given a log R and log T array, return the minimum pressure in the opacity table.
    """
    R = np.atleast_1d(R)
    T = 10**np.atleast_1d(T)

    # Make 2D grids of all (R, T) combinations
    Rgrid, Tgrid = np.meshgrid(R, T, indexing='ij')
    T6 = (Tgrid/1e6)
    rho = (10**Rgrid * T6**3) * (u.g / u.cm**3)
    P = (rho * c.k_B * Tgrid * u.K/ (u.u * 2.8)).to(u.dyne / u.cm**2) # Assume mu=2.8

    return np.min(P).value

P0 = np.maximum(get_min_P(R_grid,T_grid), P0)  # Ensure P0 is not less than the minimum R/pressure in the table

# Starting from the lowest value of log(P), integrate P using solve_ivp
print('Solving dPdtau')
start = time.time()
tau_grid = np.concatenate((np.arange(3)/3*1e-3,np.logspace(-3,1.3,30)))
sol = solve_ivp(dPdtau, [0, 20], [P0], args=(Teff,), t_eval=tau_grid, method='RK45')
Ps = sol.y[0]
Ts = T_tau(tau_grid, Teff)
# Artificially cut the deep layer temperature due to convection.
# Get new tau for later
cutoff_val = convective_cutoff * Teff
idx_cut = np.argmax(Ts >= cutoff_val)
tau_cut = tau_grid[idx_cut]
tau_grid = np.minimum(tau_grid, tau_cut)
Ts = np.minimum(Ts,convective_cutoff*Teff)


# Calculate rho values from new Ps and Ts
end = time.time()
print(f"Equilibrium Elapsed time: {end - start:.3f} seconds")
print('Calculating rhos')
nums,mu = mu_from_P_T_pairwise(Ps, Ts)
rhos = (Ps*u.dyne/u.cm**2 /(c.k_B*Ts*u.K)*u.u*mu).to(u.g/u.cm**3).value

# Interpolate onto the tau grid
print('Interpolating kappa bars')
kappa_bars = kappa_bar_l_interp((get_R(Ps, Ts), np.log10(Ts)))
kappa_bars = 10**kappa_bars # Convert from log kappa to  

# First, lets plot a continuum spectrum
print("Begin computing continuum spectrum")
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
#plt.figure(1)
plt.figure(1)
plt.clf()
tsuji_K, tsuji_masses = eos.molecules()
nmol = len(tsuji_K['mol'])
elt_names = tsuji_K['mol'].data
fig, axes = plt.subplots(
    1, 3,
    sharey='row',      # share y-axis across all plots if scales allow
    figsize=(12, 4),
    gridspec_kw={'hspace': 0, 'wspace': 0}  # no vertical gap, small horizontal gap
)
ax1, ax2, ax3 = axes[0], axes[1], axes[2]
for i, element in enumerate(elt_names):
    n_element = []
    for num in nums:
        n_element.append(num[0,-1*(nmol+1)+i])
    ax1.plot(tau_grid, n_element, label=element)
    ax2.plot(Ts, n_element, label=element)
    ax3.plot(Ps, n_element, label=element)
ax1.set_xlabel(r'$\tau$')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_ylabel(r'$n_\text{mol}$ [1/cm$^3$]')
ax2.set_xscale('linear')
ax2.set_yscale('log')
ax2.set_xlabel(r'$T$ [K]')
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_xlabel(r'$P$ [dyne/cm$^2$]')
labelLines(ax1.get_lines(), zorder=2.5)
#plt.legend()
plt.title('Molecular Number Densities')
plt.savefig(f'figures/number_density_{Teff}.pdf', dpi=200)
plt.show()

"""if plot:
plt.title('Equilibrium Solve Test')
plt.savefig('figures/equilibrium_solve_test.pdf', dpi=300)"""

# Now lets add in all lines. The Strontium line calculation is saved under "week34" if you
# want to look at that.
wavemax = 900e-9 # nm
nu0 = 3e8/wavemax
dlnu = 7e-6
Nnu = 300000
print("Computing line opacities")

# Compute the frequency grid
nu = nu0 * np.exp(dlnu * np.arange(Nnu))
wave_nm= (c.c/(nu*u.Hz)).to(u.nm).value
kappa_all_nu_bars = np.empty((Nnu, len(tau_grid)))
for i, (T, P, rho) in enumerate(zip(Ts, Ps, rhos)):
    kappa_all_nu_bars[:,i] = opac.kappa_cont(nu, np.log10(P), T) + \
        opac.weak_line_kappa(nu0, dlnu, Nnu, np.log10(P), T) + \
        opac.strong_line_kappa(nu0, dlnu, Nnu, np.log10(P), T) + opac.molecular_line_kappa(nu0, dlnu, Nnu, P, T)
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
plt.figure(4, figsize=(10,6))
plt.clf()
plt.plot(wave_nm, 4*np.pi*H_all / 1e6, label='Flux (No Molecular Lines)')
plt.title(f'M-giant Spectra Test: {Teff} K, logg={logg}, v_macro={macroturb} km/s')
plt.xlabel('Wavelength (nm)')
plt.ylabel(r'Flux (W/m$^2$/$\mu$m)')
plt.legend()
plt.savefig(f'figures/spectrum_test_T{Teff}_no_mol_lines.pdf', dpi=300)
plt.show()