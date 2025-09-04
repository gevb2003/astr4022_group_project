import numpy as np
import os
import re
from scipy.interpolate import RegularGridInterpolator

def read_opacity_table(filename):
    """
    Reads .tron file and returns log_T array, log_R array, and the table.
    Handles header and stuck-together numbers robustly.
    """
    log_R = None
    data_rows = []
    expected_cols = None
    with open(filename, 'r') as f:
        for line in f:
            # Flexible header row detection
            if 'log T' in line:
                log_R = np.array([float(num) for num in re.findall(r'-?\d+\.\d+', line)])
                expected_cols = len(log_R) + 1  # log_T + opacities
            # Data row detection: must contain at least one float and start with a float
            elif re.search(r'-?\d+\.\d+', line):
                numbers = re.findall(r'-?\d+\.\d+', line)
                if expected_cols and len(numbers) == expected_cols:
                    data_rows.append([float(num) for num in numbers])
    if not data_rows or log_R is None:
        raise ValueError("No valid data found in file.")
    data = np.array(data_rows)
    log_T = data[:, 0]      # First column is log(T)
    table = data[:, 1:]     # Remaining columns are the table
    return log_T, log_R, table

def read_abund_table(abund_filename, opac_filename, format="fraction"):
    """
    Reads abundance table from a file and puts it into a consistent format.

    Parameters
    ----------
        abund_filename : str
            Name of the abundance file to be imported.
        opac_filename : str
            Name of the opacity file used, which contains details about the metallicity.
        format : str
            Format of the abundance table. Can be "fraction" or "log-12".
    
    Returns
    -------
        X_tot : float
            Total hydrogen mass fraction.
        Y_tot : float
            Total helium mass fraction.
        Z_tot : float
            Total metal mass fraction.
        abund : np.dict
            Dictionary containing abundance values with species names as keys.
            In linear cgs units.
    """
    # Get X, Y, Z from the filename. File name is in the format [source].[X=0.x].[Z=0.z]
    # E.g. Caffau11.7.02 has X = 0.7 and Z = 0.02
    # Use OS to strip from the filename
    parts = os.path.splitext(os.path.basename(opac_filename))[0].split('.')
    X_tot = float('0.' + parts[1])
    Z_tot = float('0.' + parts[2])
    Y_tot = 1 - X_tot - Z_tot

    dtype = [('species', 'U10'), ('atomic_number', int), ('abundance', float)]
    rows = []
    with open(abund_filename, 'r', encoding='utf-16') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            species = parts[0]
            try:
                atomic_number = int(parts[1])
            except ValueError:
                continue
            abundance = float(parts[2]) if len(parts) > 2 and parts[2] not in ('', '\x00') else np.nan
            rows.append((species, atomic_number, abundance))
    data = np.array(rows, dtype=dtype)

    # Conditional treatment of abundance format
    if format == 'fraction':
        abundances = np.array([X_tot, Y_tot] + list(data['abundance'][2:] * Z_tot))
    elif format == 'log-12':
        abundances = np.array([10**(val-12) * Z_tot if not np.isnan(val) else np.nan for val in data['abundance']])
    else:
        raise ValueError("Unknown abundance table format. Current options are 'fraction' or 'log-12'.")

    abund = dict(zip(data['species'], abundances))

    # Put into a format that can be called easier
    class ret:
        def __init__(self, X_tot_, Y_tot_, Z_tot_, abund_):
            self.X_tot = X_tot_
            self.Y_tot = Y_tot_
            self.Z_tot = Z_tot_
            self.abund = abund_

    return ret(X_tot, Y_tot, Z_tot, abund)

def nearest_opac_R(T, R, log_T, log_R, table):
    """Returns the nearest exact value from the Rosseland opacity table.

    Parameters
    ----------
        T : float
            Linear temperature in Kelvin.
        R : float
            Density? Radius? Still unsure.
        log_T : np.array
            Logarithm of temperature values from the opacity table.
        log_R : np.array
            Logarithm of radius values from the opacity table.
        table : nd.array
            2D array containing the opacity values.
    
    Returns
    -------
        opac_val : float
            Nearest exact value from the opacity table.
    """
    log_T_val = np.log10(T)
    log_R_val = np.log10(R)
    T_idx = np.searchsorted(log_T, log_T_val) - 1
    R_idx = np.searchsorted(log_R, log_R_val) - 1
    opac_val = table[T_idx, R_idx]
    return opac_val


def interp_opac_R(T, R, log_T, log_R, table):
    """
    Parameters
    ----------
        T : float
            Linear temperature in Kelvin.
        R : float
            Density? Radius? Still unsure.
        log_T : np.array
            Logarithm of temperature values from the opacity table.
        log_R : np.array
            Logarithm of radius values from the opacity table.
        table : nd.array
            2D array containing the opacity values.
    
    Returns
    -------
        opac_val : float
            Nearest exact value from the opacity table.
    """
    interpolator = RegularGridInterpolator((log_T, log_R), table)
    log_T_val = np.log10(T)
    log_R_val = np.log10(R)
    opac_val = interpolator((log_T_val, log_R_val))
    return opac_val