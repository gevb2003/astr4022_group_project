"""
TODO:
1) Output a complete table for convective star student use (not including M
dwarfs)

"""
import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as c
import astropy.units as u
import scipy.optimize as op
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import astropy.io.fits as pyfits
from astropy.table import Table
#For fsolve - no idea why this happens at high temperatures...
import warnings
plt.ion()
warnings.filterwarnings('ignore', 'The iteration is not making good progress')
tsuji_K = Table.read('../data/tsuji_K.dat', format='ascii.fixed_width_no_header', \
 names=('mol', 'c0', 'c1', 'c2','c3','c4','molcode','ediss','comment'), col_starts=(0,7,19,31,43,55,67,80,87))

debroglie_const = (c.h**2/2/np.pi/c.m_e/c.k_B).cgs.value
eV_Boltzmann_const = (u.eV/c.k_B).cgs.value
deybe_const = (c.k_B/8/np.pi/c.e.esu**2).cgs.value
delchi_const = (c.e.esu**2/(1*u.eV)).cgs.value

def solarmet():
    """Return solar metalicity abundances by number and masses for low mass elements.
    From Asplund ete al (2009), up to an abundance of 1e-5 only plus Na, K, Ca. 
    Degeneracy and ionization energies from IA05 in Scholz code
    
    H++ is actually H-, and the code treats this appropriately as a special case"""
    elt_names = np.array(['H', 'He',   'C',   'N',   'O',   'Ne',  'Na',  'Mg',  'Si',  'S',   'K',   'Ca',  'Fe',  'Ti'])
    n_p =        np.array([1,   2,      6,     7,     8,     10,    11,    12,    14,    16,    19,    20,    26,     22])
    masses=      np.array([1.0, 4.0,    12.01, 14.01, 16.00, 18.0,  22.99, 24.31, 28.09, 32.06, 39.10, 40.08, 55.85, 47.9])
    abund = 10**(np.array([12, 10.93,   8.43,  7.83,  8.69,  7.93,  6.24,  7.60,  7.51,  7.12,  5.03,  6.34,  7.50, 4.95])-12)
    ionI  =      np.array([13.595,24.58,11.26, 14.53, 13.61, 21.56, 5.14,  7.644, 8.149, 10.357,4.339, 6.111, 7.87, 6.82])
    ionII  =     np.array([-0.754,  54.403,  24.376,29.593,35.108,40.96, 47.29, 15.03, 16.34, 23.40, 31.81, 11.87, 16.18, 13.57])

    #Degeneracy of many of these elements are somewhat temperature-dependent,
    #as it is really a partition function. But as this is mostly H/He plus 
    #other elements as a mass reservoir and source of low-T
    #electrons, we're ignoring this. 
    gI =   np.array([2,1,9,4,9,1,2,1,9,9,2,1,25,21])
    gII =  np.array([1,2,6,9,4,6,1,2,6,4,1,2,30,28])
    
    #A lot of these degeneracies are educated guesses! But we're not worried
    #about most elements in the doubly ionized state. 
    gIII = np.array([1,1,1,6,9,9,6,1,1,9,6,1,30,1])
    return abund, masses, n_p, ionI, ionII, gI, gII, gIII, elt_names
    
def equilibrium_equation(rho, T):
    """Find the components of the chemical equilibrium equation in matrix form.
    
    There are two parts - linear in partial pressures and logarithmic in 
    partial pressures. 
    """
    #Input constants and abundances
    abund, masses, n_p, ionI, ionII, gI, gII, gIII, elt_names = solarmet()
    
    #Find the reference pressure, which is the ideal gas pressure corresponding to
    #atomic Hydrogen only.
    log_P_ref = np.log10( (rho*c.k_B*T/u.u).to(u.dyn/u.cm**2).value )
    nmol = len(tsuji_K)
    natom = len(abund)
    
    #The linear matrix, with one equation (row) per atom,  one
    #for the electron.
    linear_matrix = np.zeros((natom+1, 1 + 2*natom + nmol))
    linear_b = np.zeros((natom + 1))
    
    #The logarithmic matrix, with one equation per ion and one
    #for each molecule
    log_matrix = np.zeros((natom + nmol, 1 + 2*natom + nmol))
    log_b = np.zeros((natom + nmol))
    
    #The theta value for computing the molecular equilibrium constants
    th = 5040/T.to(u.K).value
    #The equivalent for the  atoms
    eV_kTln10 = float(1*u.eV/c.k_B/T/np.log(10))
    
    #First, the electron equation
    linear_matrix[0,1+natom:1+2*natom] = 1
    linear_matrix[0,0] = -1
    
    #Next, the RHS of the equation.
    linear_b[1:] = abund/np.sum(masses*abund)
    #Now, the atomic and ion part of the equation
    for i in range(natom):
        linear_matrix[i+1,i+1] = 1
        linear_matrix[i+1,natom+i+1] = 1
        
    #The logarithmic part of the matrix for atoms.
    debroglie=np.sqrt(debroglie_const/T.to(u.K).value)
    kBT = (c.k_B*T).cgs.value
    for i in range(natom):
        log_b[i] = np.log10(2*kBT/debroglie**3 *gII[i]/gI[i]) - eV_kTln10*ionI[i]
        log_matrix[i,0] = 1
        log_matrix[i,1+i] = -1
        log_matrix[i,1+i+natom] = 1
        
    #Next, the molecular part of the equation
    for i in range(nmol):
        log_matrix[natom + i, 2*natom+1+i] = -1
        mol = tsuji_K[i]
        log_b[natom + i] = mol['c0'] + mol['c1']*th + mol['c2']*th**2 + mol['c3']*th**3 + mol['c4']*th**4
        for j in range(int(mol['molcode'][0])):
            atom = int(mol['molcode'][1+j*3:3+j*3])
            natom_in_mol = int(mol['molcode'][3+j*3:4+j*3])
            k = np.argwhere(n_p==atom)[0,0]
            linear_matrix[k+1,2*natom+1+i] = natom_in_mol
            log_matrix[natom + i, 1+k] = natom_in_mol

    return linear_matrix, linear_b, log_matrix, log_b, log_P_ref
    
def eq_solve_func(logps, linear_matrix, linear_b, log_matrix, log_b, log_P_ref, abund):
    """Equation to put into scipy solve.
    
    Returns:
    --------
    resid: np.array
        concatenated log resids (n_atoms + n_mol) then concatenated linear resids 
        (1 + n_atoms)
    """
    ps = ( 10**(logps-log_P_ref) )
    log_part    = np.dot(log_matrix, logps) - log_b
    linear_part = np.dot(linear_matrix, ps) - linear_b 
    #Dividing the linear part by the abundance just makes all numbers near 0 and the 
    #same order of magnitude.
    linear_part[1:] /= abund
    return np.concatenate((log_part, linear_part))
    
def equilibrium_solve(rho, T, plot=False):
    """Solve for atomic and molecular equilibrium. 
    
    Parameters
    ----------
    rho:  
         Dimensioned density
    T:
        Dimensioned temperature
    
    Returns:
    --------
    log(pressure) in dyn/cm^2 for:
        e-, all neutrals A (13), ions A+ (13) elements, 
        then pressure for each molecule (4), then finally H-
        
    """
    linear_matrix, linear_b, log_matrix, log_b, log_P_ref = equilibrium_equation(rho, T)
    abund, masses, n_p, ionI, ionII, gI, gII, gIII, elt_names = solarmet()
    
    #Start with almost no electrons or molecules, and atoms in proportion to 
    #their abundances.
    #x0 = np.ones(linear_matrix.shape[1])*log_P_ref - 12
    #x0[1:len(abund)+1] = log_P_ref + np.log10(abund)
    
    #Starting point - ignore atoms.
    n_e, ns, mu, Ui = ns_from_rho_T(rho,T)
    
    #Convert number density to log partial pressure.
    if min(ns)<0:
        import pdb; pdb.set_trace()
    logp_atom = np.log10((ns*c.k_B*T).to(u.dyn/u.cm**2).value)
    logpe = np.log10((n_e*c.k_B*T).to(u.dyn/u.cm**2).value)
    x0 = np.ones(linear_matrix.shape[1])*log_P_ref - 12
    
    x0[0] = logpe #n_e
    x0[1+np.arange(len(abund))] = logp_atom[3*np.arange(len(abund))] #neutrals
    x0[len(abund)+1+np.arange(len(abund))] = logp_atom[3*np.arange(len(abund))+1] #ions
    
    #Deplete H, C, O, N, Ti just a little to stop the algorithm getting stuck.
    x0[1] -= 0.4
    x0[3] -= 0.2
    x0[4] -= 0.2
    x0[5] -= 0.4
    x0[14] -= 0.3
    
    #Also start the most abundant molecules as 0.5 dex less than their limiting constituent
    x0[-4] = x0[1]-0.5
    x0[-3] = x0[3]-0.5
    x0[-2] = x0[5]-0.5
    x0[-1] = x0[5]-0.5
    
    #Now solve for the abundances of the molecules!
    res = op.root(eq_solve_func, x0, args=(linear_matrix, linear_b, log_matrix, log_b, log_P_ref, abund),method='lm')#, )
    
    #For testing, we can make a plot here!
    if plot:
        natom = len(abund)
        nmol = len(tsuji_K['mol'])
        plt.clf()
        plt.xticks(np.arange(natom + nmol), np.concatenate((elt_names, tsuji_K['mol'].data)))
        plt.plot(x0[1:natom+1], 'ro')
        plt.plot(x0[natom+1:], 'go')
        plt.plot([x0[0]], 'rs')
        plt.plot(res.x[1:natom+1],'r', label='Neutral solution')
        plt.plot(res.x[natom+1:],'g', label='Ionised solution')
        plt.plot([res.x[0]], 'kx', label='Electron solution')
        plt.plot([logp_atom[2]], 'ko', label='H-')
        plt.legend()
        if (np.min(res.x) < -20):
            plt.ylim([-25,np.max(res.x)+0.5])
        plt.ylabel(r'log$_{10}$(p) (dyn/cm$^2$)')
    
    #import pdb; pdb.set_trace()
    #test = eq_solve_func(res.x, linear_matrix, linear_b, log_matrix, log_b, log_P_ref, abund)
    if res.success:
        return np.concatenate( (res.x, [logp_atom[2]]))
    else:
        import pdb; pdb.set_trace()

def ion_mass_g(masses, gI, gII, gIII, max_ionize=2):
    if max_ionize != 2:
            print("Not fully implemented!")
    m_ions = np.empty(len(masses)*3-1)
    ig = np.empty(len(masses)*3-1)
    m_ions[:2] = masses[0]
    m_ions[2:] = np.repeat(masses[1:],3)
    ig[0] = gI[0]
    ig[1] = gII[0]
    ig[2 + 3*np.arange(len(masses)-1)] = gI[1:]
    ig[3 + 3*np.arange(len(masses)-1)] = gII[1:]
    ig[4 + 3*np.arange(len(masses)-1)] = gIII[1:]
    return m_ions, ig

def electron_pressure(n_e, T):
    """Compute the electron pressure for a partly degenerate, partly relativistic 
    equation of state. Unfortunately, this needs an integral.
    !!! For now, neglect degeneracy !!!"""
    #Momentum step based on temperature.
    #pstep = 0.2*sqrt(c.m_e*c.k_b*T)
    #Maximum momentum.
    #maxp = np.sqrt(np.max([(mu + c.m_e*c.c^2 + 8*c.k_b*T)^2-c.m_e^2*c.c**4,0]))/c
    return (n_e*c.k_B*T).cgs
    
def simplified_eos_rho_T(rho, T, ionised=True):
    """Assume that H and He are ionised. Return the gas pressure for this density and
    the adiabatic gamma
    
    Parameters
    ----------
    rho: Density, including units from astropy units.
    T: Temperature, including units from astropy units.
    
    Returns
    -------
    Pressure, including units from astropy.units
    Adiabatic index (dimensionless)
    """
    #Input the abundances of the elements
    abund, masses, n_p, ionI, ionII, gI, gII, gIII, elt_names  = solarmet()
    
    #Find the number density of H
    n_h = rho/(np.sum(abund*masses)*u.u)
    
    #Assume that H and He are totally ionized. Ignore heavier elements.
    if ionised:
        n_e = n_h*(abund[0] + 2*abund[1])
    else:
        n_e = n_h*0
    
    #Now find total pressure.
    P = ((n_h*np.sum(abund) + n_e)*c.k_B*T).cgs
    
    return P, 5/3, 5/3, 0*u.erg/u.g
    
def saha(n_e, T):
    """Compute the solution to the Saha equation as a function of electron number
    density and temperature, in CGS units. 
    
    This enables the problem to be a simple Ax=b linear problem.
    Results from this function can be used to solve the Saha equation as e.g. a function 
    of rho and T via e.g. tabulating or solving.
    
    Parameters
    ----------
    n_e: double
        the electron number density in cm^{-3}
    T: double
        Temperature in K.
    
    Returns
    -------
    rho: astropy.units quantity compatible with density
    mu: Mean molecular weight (dimensionless, i.e. to be multiplied by the AMU)
    ns: A vector of number densities of H, H+, H-, He, He+, He++ in cm^{-3}
    
    """
    
    #Input the abundances of the elements
    abund, masses, n_p, ionI, ionII, gI, gII, gIII, elt_names  = solarmet()
    n_elt = len(n_p)
    
    #Find the Deybe length, and the decrease in the ionization potential in eV, 
    #according to Mihalas 9-178
    deybe_length = np.sqrt(deybe_const*T/n_e)
    z1_delchi = delchi_const/deybe_length
    
    #This will break for very low temperatures. In this case, fix a zero  
    #ionization fraction
    if (T<1000):
        ns = np.zeros(n_elt*3)
        ns[3*np.arange(n_elt)] = abund
        ns = np.maximum(n_e*1e15*ns, 1e-300)
    else:
        #The thermal de Broglie wavelength. See dimensioned version of 
        #this constant above
        debroglie=np.sqrt(debroglie_const/T)
    
        #Hydrogen ionization. We neglect the excited states because
        #they are only important when the series diverges... 
        #!!! h1  = 2./debroglie**3 *gII[0]/gI[0]*np.exp(-(ionI[0] - n_p[0]*z1_delchi)*eV_Boltzmann_const/T)  
    
        #Now construct our matrix of n_elt*3-1 equations defining these number densities.
        A = np.zeros( (3*n_elt,3*n_elt) );
        for i in range(n_elt):
            #Our first equation for this element: the ratio of the sum of element number
            #densities to H number densities is the ratio of abundances.
            A[3*i,  3*i:3*(i+1)] = [-abund[0],-abund[0],-abund[0]]
            A[3*i,  :3] = [abund[i],abund[i],abund[i]]
            
            #Element single ionization. 
            he1 = 2./debroglie**3 *gII[i]/gI[i]*np.exp(-(ionI[i] - n_p[i]*z1_delchi)*eV_Boltzmann_const/T)
            #The second equation for this element: 
            A[3*i+1,3*i:3*i+2]=[-he1/n_e, 1]
                
            #Element double-ionization, or H-
            if i == 0:
                he2 = 2./debroglie**3 *gI[i]/gIII[i]*np.exp(-(np.abs(ionII[i]) - n_p[i]*z1_delchi)*eV_Boltzmann_const/T)
                A[3*i+2,3*i:3*i+3]  =[1,0,-he2/n_e]   
            else:
                #The same as the single ionization formula! 
                he2 = 2./debroglie**3 *gIII[i]/gII[i]*np.exp(-(ionII[i] - n_p[i]*z1_delchi)*eV_Boltzmann_const/T)
                A[3*i+2,3*i+1:3*i+3]  =[-he2/n_e, 1]

        #Add in the equation for electron number density. This over-writes the equation
        #n_H + n_H  
        A[0,:] =np.concatenate(([0,1,-1], np.tile([0,1,2], n_elt-1)))
        
        #Now solve Ax=b, where b only has one non-zero number (the electron 
        #number density)
        b =np.zeros((3*n_elt))
        b[0] = n_e
        ns = np.linalg.solve(A,b)
        
        #Make sure that all number densities are positive.
        ns = np.maximum(ns,1e-300) 
        #import pdb; pdb.set_trace()
    
    #The next lines ensure ionization at high electron pressure, roughly due to nuclei 
    #being separated by less. than the size of an atom. 
    #There is also a hack included based on a typical atom size.
    ns_highT = np.zeros(n_elt*3)
    ns_highT[1] = abund[0]
    ns_highT[2 + np.arange(n_elt-1)*3] = abund[1:]
    ns_highT=ns_highT/(abund[0]+2*np.sum(abund[1:]))*n_e
    atom_size = 1e-8 #In cm
    if (n_e*atom_size**3 > 2):
        ns=ns_highT
        print("High T")
    elif (n_e*atom_size**3 > 1):
        frac=((n_e*atom_size **3) - 1)/1.0
        ns = frac*ns_highT + (1-frac)*ns
                
    #For normalization... we need the number density of Hydrogen
    #nuclei, which is the sum of the number densities of H and H+ and H-
    n_h = np.sum(ns[:3])
    
    #Density. Masses should be scalars.
    rho_cgs = n_h*np.sum(abund*masses)*c.u.to(u.g).value
   
    #Fractional "abundance" of electrons.
    f_e = n_e/n_h
    
    #mu is mean "molecular" weight, and we make the approximation that
    #electrons have zero weight.
    mu = np.sum(abund*masses)/(np.sum(abund) + f_e)
    
    #Finally, we should compute the internal energy with respect to neutral gas.
    #This is the internal energy per H atom, divided by the mass in grams per H atom. 
    Ui=(ns[1]*13.6 + ns[4]*24.58 + ns[5]*(54.403+24.58))*u.eV/n_h/np.sum(abund*masses*u.u);
    
    return rho_cgs, mu, Ui, ns
    
def saha_solve(log_n_e_mol_cm3, T, rho_0_in_g_cm3):
    """Dimensionless version of the Saha equation routine, to use in np.solve to
    solve for n_e at a fixed density."""
    n_e = np.exp(log_n_e_mol_cm3[0])*c.N_A.value
    rho, mu, Ui, ns = saha(n_e, T)
    
    return np.log(rho_0_in_g_cm3/rho)

def saha_solve_P(log_n_e_mol_cm3, P_0_cgs, T):
    """Dimensionless version of the Saha equation routine, to use in np.solve to
    solve for n_e at a fixed density."""
    n_e = np.exp(log_n_e_mol_cm3[0])*c.N_A.value
    rho, mu, Ui, ns = saha(n_e, T)
    #Ideal gas equation...
    P = rho/mu*(c.k_B*T*u.K/u.u).cgs.value
    return np.log(P_0_cgs/P)
 
def ns_from_rho_T(rho,T):
    """Compute number densities given a density and temeprature"""
    
    rho_in_g_cm3 = rho.to(u.g/u.cm**3).value
    
    #Start with the electron number density equal in mol/cm^3 equal to the density
    #in g/cm^3, or a much lower number at low temperatures. Modify it a little to help
    #starting point at low T
    #FIXME: This stopped working well once H- was added.
    x0 = np.log(rho_in_g_cm3)
    T_K = np.maximum(T.to(u.K).value,1000)
    x0 += np.log(2/(10*np.exp(40e3/T_K) + 1))
    
    #The following line is the important one that can't have units associated
    #with it, as it takes too long.
    res = op.fsolve(saha_solve, x0, args=(T.cgs.value, rho_in_g_cm3), xtol=1e-6)
    n_e = np.exp(res[0])*c.N_A.value
    rho_check, mu, Ui, ns = saha(n_e, T.cgs.value)
    if (np.abs(rho_check/rho_in_g_cm3-1) > 0.01):
        raise UserWarning("Density check incorrect!")

    #Return dimensioned quantities
    return n_e*(u.cm**(-3)), ns*(u.cm**(-3)), mu, Ui 

def ns_from_P_T(P,T):
    """Compute number densities given a pressure and temperature"""
    
    P_cgs = P.to(u.dyne/u.cm**2).value
    
    #Same as ns_from_rho_T, at a mu=1.
    x0 = np.log(P_cgs/((c.k_B*T*u.K/u.u).cgs.value))
    T_K = np.maximum(T.to(u.K).value,1000)
    x0 += np.log(2/(10*np.exp(40e3/T_K) + 1))
    
    #The following line is the important one that can't have units associated
    #with it, as it takes too long.
    res = op.fsolve(saha_solve_P, x0, args=(P_cgs, T.cgs.value), xtol=1e-6)
    n_e = np.exp(res[0])*c.N_A.value
    rho, mu, Ui, ns = saha(n_e, T.cgs.value)
    
    #Return dimensioned quantities
    return n_e*(u.cm**(-3)), ns*(u.cm**(-3)), mu, Ui, rho

def compute_entropy(rho, T):
    """Compute the specific entropy for a gas mixture of density rho 
    and temperature T"""
    #Find number densities
    n_e, ns, _, _ = ns_from_rho_T(rho, T)
    
    #Now that we have the number densities of all ions, we can compute 
    #their de Broglie wavelengths. 
    abund, masses, n_p, ionI, ionII, gI, gII, gIII, elt_names = solarmet()
    imass, ig = ion_mass_g(masses, gI, gII, gIII)
    
    #Add in the electron with spin 1/2
    mion = np.concatenate(([c.m_e], imass*u.u))
    gion = np.concatenate(([2], ig))
    debroglie = (c.h/np.sqrt(2*np.pi*mion*c.k_B*T)).cgs
    #Number density of baryons
    n_b = (rho/u.u).cgs
    ns = np.maximum(ns, 1e-3*u.cm**(-3))
    ns = np.concatenate(([n_e],ns))
    fi = ns/n_b
    specific_entropy = np.sum(fi*(np.log(1/ns/debroglie**3) + np.log(gion))) + 2.5*np.sum(fi)   
    
    return specific_entropy.value
    
def eos_rho_T(rho, T, full_output=False):
    """Compute the key equation of state parameters via the Saha equation
    
    Parameters
    ----------
    rho: astropy.units quantity compatible with density
    T: astropy.units quantity gas Temperature
    
    Returns
    -------
    P: Gas Pressure
    n_e: Electron number density
    ns: Number densities of H, H+, He, He+, He++
    mu: Mean molecular weight in atomic mass units
    Ui: Internal energy per unit mass due to ionization
    """
    if (rho.ndim>0) or (T.ndim>0):
        raise ValueError("Can not input array - single values only.")
        
    if T<3000*u.K:
        return simplified_eos_rho_T(rho, T, ionised=False)
    
    #Compute number densities
    n_e, ns, mu, Ui  = ns_from_rho_T(rho,T)
        
    #The total gas pressure is just the sum of the number densities multiplied by kT
    #!!! We should take heavier elements into account here as well, and wrap this in 
    #a function. Maybe simpler just to add in more elements to Saha?
    P = (np.sum(ns)*c.k_B*T).to(u.dyn/u.cm**2)
    P += electron_pressure(n_e, T)
        
    #Next, find the adaibatic exponent. As we are neglecting radiation, pressure, there
    #is only a single gamma (e.g. https://ui.adsabs.harvard.edu/abs/2002ApJ...581.1407S/abstract)
    #We will do this two different ways, to double-check. I'm double checking because 
    #the results didn't seem to exactly match Unsold's 1968 book:
    #Physik der Sternatmosphaeren MIT besonderer Beruecksichtigung der Sonne, which is 
    #referenced in e.g.
    #http://www.ifa.hawaii.edu/users/kud/teaching/4.Convection.pdf
    
    #In both cases, we need to numerically compute derivatives. We do this by slightly
    #increasing temperature and density, and re-calculating.
    dlog = 1e-4
        
    #Increase temperature
    n_e_Tplus, ns_Tplus, mu_Tplus, Ui_Tplus = ns_from_rho_T(rho,T*np.exp(dlog))
    P_Tplus = (np.sum(ns_Tplus)*c.k_B*T*np.exp(dlog)).to(u.dyn/u.cm**2)
    P_Tplus += electron_pressure(n_e_Tplus, T*np.exp(dlog))
    
    #Increase Density
    n_e_rhoplus, ns_rhoplus, mu_rhoplus, Ui_rhoplus = ns_from_rho_T(rho*np.exp(dlog),T)
    P_rhoplus = (np.sum(ns_rhoplus)*c.k_B*T).to(u.dyn/u.cm**2)
    P_rhoplus += electron_pressure(n_e_rhoplus, T)
    
    #Compute the 4 logarithmic derivatives. We scale internal energy by rho/P to make
    #it dimensionless.
    dUidlrho_scaled = float( (Ui_rhoplus - Ui)/dlog*rho/P )
    dUidlT_scaled = float( (Ui_Tplus - Ui)/dlog*rho/P )
    dlPdlrho = float( (P_rhoplus - P)/dlog/P )
    dlPdlT = float( (P_Tplus - P)/dlog/P )
    
    #Now the tricky bit. We have to use partial derivative relations to move from 
    #Ui(rho, T) to Ui(rho, P), which we call UU. 
    #To be more accurate, we strictly need to repeat this exercise for U3(P, T) in order 
    #to correctly take into account the change of mean molecular weight. Note that 
    #section 3 of Stothers (2002) assumed a constant ionisation state.
    dUUdlrho_scaled = dUidlrho_scaled - dUidlT_scaled * (dlPdlrho/dlPdlT)
    dUUdlP_scaled = dUidlT_scaled / dlPdlT

    #The following comes directly from the definition of adiabatic, from e.g. the derivation
    #on page 5 of https://websites.pmc.ucsc.edu/~glatz/astr_112/lectures/notes6.pdf
    gamma1 = (5/2 - dUUdlrho_scaled) / (3/2 + dUUdlP_scaled)
    
    # For method 2, see equation 18.8 on page 571 of Cox and Guili. 
    #We need an additional two normalised logarithmic derivatives. 
    #They are all zero in the absence of a phase change
    dlmudlT = float( (mu_Tplus - mu)/mu/dlog )
    dlmudlrho = float( (mu_rhoplus - mu)/mu/dlog )
    dUidlT_scaled = float( (Ui_Tplus - Ui)/dlog*mu*u.u/c.k_B/T )
        
    #Composition quantities to match Cox and Guili's equations. As far as I can tell,
    #this gamma is identical to the gamma above, so a great check.
    chi_T    = 1 - dlmudlT
    chi_rho  = 1 - dlmudlrho
    gamma1_old = chi_rho + chi_T**2/(3/2*chi_T + dUidlT_scaled)
    
    #Now, to compute the other gammas, use 3.96, 3.98 and 3.99 in Hansen, Kawaler and
    #Trimble (or find Cox and Guili in the library again!)
    nabla_ad = (gamma1 - chi_rho)/gamma1/chi_T
    gamma2 = 1/(1-nabla_ad)
    gamma3 = 1 + gamma1*(gamma2 - 1)/gamma2 
    
    if (np.abs(gamma1 - gamma1_old) > 0.05) and (T < 1e6*u.K):
        import pdb; pdb.set_trace()
    
    if full_output:
        return P, n_e, ns, mu, Ui, gamma1, gamma2, gamma3, gamma1_old 
    else:
        return P, gamma1, gamma3, Ui

def adiabatic_logTderiv(logrho_cgs, logT_K):
    """
    For the purpose of computing an adiabatic profile, compute the 
    derivative of log(T)/dlog(rho)

    Parameters
    ----------
    logrho_cgs : float
        log10(density_in_g).
    logT_K : float
        log10(temperatue in Kelvin).

    Returns
    -------
    list
        [Adiabatic Gamma_3].

    """
    P, gamma1, gamma3, Ui = eos_rho_T((10**logrho_cgs)*u.g/u.cm**3, (10**logT_K[0])*u.K)
    return [gamma3-1]

def s_from_lT_rho(lT, rho, starg):
    """Compute s from log10(T) and rho"""
    return compute_entropy(rho, (10**lT[0])*u.K) - starg

def T_from_rho_s(rho, s):
    #To solve for temperature, we need an initial guess. 
    x0 = 6 + 0.4*np.log10(rho.to(u.g/u.cm**3).value) + 0.4*(s-16.7)/np.log(10)
    res = op.fsolve(s_from_lT_rho, x0, args=(rho, s), xtol=1e-6)
    return (10**res[0])*u.K

def rho_s_tables(rhos, ss, savefile=''):
    """
    For an array of densities and specific entropies, create tables of
    Pressure, gamma_1 and Temperature.
    
    Parameters
    ----------
    rhos : array
        input density list.
    s2 : array
        input entropy list.

    Returns
    -------
    Pressure, gamma_1 and Temperature tables.

    """
    rhos_log = np.log(rhos.to(u.g/u.cm**3).value)
    nrho = len(rhos)
    ns = len(ss)
    P_tab = np.empty((ns, nrho))
    T_tab = np.empty((ns, nrho))
    gamma1_tab = np.empty((ns, nrho))
    delta_log = 1e-4
    for i, s in enumerate(ss):
        for j, rho in enumerate(rhos):
            T_tab[i,j] = T_from_rho_s(rho, s).to(u.K).value
            #Compute number dnsities
            n_e, ns, mu, Ui  = ns_from_rho_T(rho,T_tab[i,j]*u.K)
            #Compute pressre
            P = (np.sum(ns)*c.k_B*T_tab[i,j]*u.K).to(u.dyn/u.cm**2)
            P += electron_pressure(n_e, T_tab[i,j]*u.K)
            P_tab[i,j] = P.value
        #Now we create a lnP vs ln(rho) function
        f = interp1d(rhos_log, np.log(P_tab[i]), fill_value='extrapolate')
        gamma1_tab[i] = (f(rhos_log+delta_log) - f(rhos_log-delta_log))/2/delta_log
        
    if len(savefile)>0:
            hdu1 = pyfits.PrimaryHDU(T_tab)
            hdu1.header['CRVAL1'] = (rhos_log[0])/np.log(10)
            hdu1.header['CDELT1'] = (rhos_log[1]-rhos_log[0])/np.log(10)
            hdu1.header['CTYPE1'] = 'log10(density) [g/cm^3]'
            hdu1.header['CRVAL2'] = ss[0]
            hdu1.header['CDELT2'] = ss[1]-ss[0]
            hdu1.header['CTYPE2'] = 'entropy [k_B/baryon]'
            hdu1.header['EXTNAME'] = 'T [K]'
            hdu2 = pyfits.ImageHDU(P_tab)
            hdu2.header['EXTNAME'] = 'P [dyn/cm**2]'
            hdu3 = pyfits.ImageHDU(gamma1_tab)
            hdu3.header['EXTNAME'] = 'Gamma1'
            hdulist = pyfits.HDUList([hdu1, hdu2, hdu3])
            hdulist.writeto(savefile, overwrite=True)
    return P_tab, gamma1_tab, T_tab

def P_T_tables(Ps, Ts, savefile=''):
    """
    For an array of densities and specific entropies, create tables of
    density (TODO: gammas and entropy).
    
    Pressures are assumed in a logarithmic grid.
    
    Parameters
    ----------
    Ps : array
        input pressure list.
    Ts : array
        input temperature list.

    Returns
    -------
    density, internal energy, number densities tables

    """
    Ps_log = np.log(Ps.to(u.dyne/u.cm**2).value)
    nP = len(Ps)
    nT = len(Ts)
    rho_tab = np.empty((nP, nT))
    mu_tab = np.empty((nP, nT))
    Ui_tab = np.empty((nP,nT))
    Q_tab = np.empty((nP,nT))
    cP_tab = np.empty((nP,nT))
    
    #Find the number of atoms and ions
    n_e, ns, mu, Ui, rho  = ns_from_P_T(Ps[0], Ts[0])
    n_species = len(ns)
    
    ns_tab = np.empty((nP,nT,n_species))
    n_e_tab = np.empty((nP,nT))
    
    #gamma1_tab = np.empty((nT, nP))
    #entropy_tab = np.empty((nT, nP))
    dT = 1*u.K #A small amount of temperature!
    k_b_u_cgs = (c.k_B/c.u).cgs.value
    for i, P in enumerate(Ps):
        for j, T in enumerate(Ts):
            #Compute number densities and densities, and also a single-sided derivative
            n_e, ns, mu, Ui, rho  = ns_from_P_T(P, T)
            _, _, mu_plus, Ui_plus, rho_plus = ns_from_P_T(P, T + dT)
            
            #Fill in the tables not involving derivatives
            rho_tab[i,j] = rho #Already in cgs
            Ui_tab[i,j] = Ui.to(u.erg/u.g).value
            mu_tab[i,j] = mu
            ns_tab[i,j] = ns.cgs.value
            n_e_tab[i,j] = n_e.cgs.value
            
            #Now the tables involving derivatives
            Q_tab[i,j] = 1 - (mu_plus - mu)/dT * T/mu
            cP_tab[i,j] = ((Ui_plus - Ui)/dT).cgs.value + 5/2*k_b_u_cgs*Q_tab[i,j]/mu_tab[i,j] 

        
    if len(savefile)>0:
            hdu1 = pyfits.PrimaryHDU(rho_tab)
            hdu1.header['CRVAL1'] = Ts[0].cgs.value
            hdu1.header['CDELT1'] = (Ts[1]-Ts[0]).cgs.value
            hdu1.header['CTYPE1'] = 'Temperature [K]'
            hdu1.header['CRVAL2'] = (Ps_log[0])/np.log(10)
            hdu1.header['CDELT2'] = (Ps_log[1]-Ps_log[0])/np.log(10)
            hdu1.header['CTYPE2'] = 'log10(pressure) [dyne/cm^2]'
            hdu1.header['EXTNAME'] = 'rho [g/cm**3]'
            hdu2 = pyfits.ImageHDU(Ui_tab)
            hdu2.header['EXTNAME'] = 'Ui [erg/g]'
            hdu3 = pyfits.ImageHDU(mu_tab)
            hdu3.header['EXTNAME'] = 'mu'
            hdu4 = pyfits.ImageHDU(ns_tab)
            hdu4.header['EXTNAME'] = 'ns [cm^-3]'
            hdu5 = pyfits.ImageHDU(n_e_tab)
            hdu5.header['EXTNAME'] = 'n_e [cm^-3]'
            hdu6 = pyfits.ImageHDU(Q_tab)
            hdu6.header['EXTNAME'] = 'Q'
            hdu7 = pyfits.ImageHDU(cP_tab)
            hdu7.header['EXTNAME'] = 'cP [erg/K/g]'
            hdulist = pyfits.HDUList([hdu1, hdu2, hdu3, hdu4, hdu5,hdu6,hdu7])
            hdulist.writeto(savefile, overwrite=True)
    return rho_tab, Ui_tab, mu_tab, ns_tab
        

if __name__=='__main__':
	#Here are a series of simple standalone tests, which you can run if you want!
	
	#Test 1: Chemical equilibrium for a Proxima Cen like atmosphere.
    if False: #Find a low temperature chemical equilibrium.
        Ts = (np.array([1500,1660,1930,2350,3320]))*u.K
        Pg = np.array([2e3,1e4,1e5,1e6,1e7])*u.dyn/u.cm**2 
        #A rough density based on a mean molecular weight of 1.5
        rhos = (Pg/(c.k_B*Ts)*u.u*1.5).to(u.g/u.cm**3) 
        #linear_matrix, linear_b, log_matrix, log_b, log_P_ref = equilibrium_equation(rho, T)
        logPs = []
        for rho, T in zip(rhos, Ts):
            print("Doing temperature: {:.1f}".format(T))
            logPs += [equilibrium_solve(rho, T, plot=True)]
            plt.pause(.5)
        logPs = np.array(logPs)
   
    #Test 2: Stellar interiors
    if False:
        #Gamma3-1 is d log(T) / dlog(rho). We start with 1g/cm**3 and a 
        #temperature of 4 MK, and use Gamma3 to solve the differential
        #equation.
        res = solve_ivp(adiabatic_logTderiv, np.log10([1e1, 1e-7]), [np.log10(4e6)], method='RK23', max_step=0.5)
        print("Computed profile...")
        plt.clf()
        plt.loglog(10**(res.t), 10**(res.y[0]),'.')
        plt.xlabel('Density(g/cm^3)')
        plt.ylabel('Temperature(K)')
        spec_e = []
        for rhol, Tl in zip(res.t, res.y[0]):
            spec_e += [compute_entropy((10**rhol)*u.g/u.cm**3, (10**Tl)*u.K)]
    
        print("Computed profile entropies")
    
    #Test 3: Stellar interiors
    if False:
        rhos = np.logspace(-9,3,27)*u.g/u.cm**3
        Ts = []
        s = 20
        for rho in rhos:
            Ts += [T_from_rho_s(rho, s).to(u.K).value]
        plt.loglog(rhos.value, Ts)
        print("Computed profile based on entropy")
    
    
        P_tab, gamma1_tab, T_tab = rho_s_tables(rhos, 12 + np.arange(11), savefile='adiabats.fits')
        for i in range(len(T_tab)):
            plt.plot(rhos.value, T_tab[i])
            
    #Test 4: Make an equation of state table
    if True:
        Ps = np.logspace(-4,6,41)*u.dyn/u.cm**2
        Ts = (2000 + 500*np.arange(47))*u.K
        P_T_tables(Ps, Ts, savefile='rho_Ui_mu_ns_ne_Q_cP.fits')
        
        
        