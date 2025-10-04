# This script reads the .states, .trans, .pf and .broad files from exomol datatbase for a given molecule (TiO, VO, CN, CO, H2O)
import os
import requests
import pandas as pd
from io import StringIO
from io import BytesIO
import bz2
import numpy as np


# List of wavenumber ranges for molecles with more than one transfile
H2O_trans_range = np.arange(0, 41201, 100)
VO_trans_range = np.arange(0, 45001, 500)

# Convert to padded string to identify files
H2O_str = [f"{num:05d}" for num in H2O_trans_range]
VO_str = [f"{num:05d}" for num in VO_trans_range]


MOLECULE_URLS = {
    'TiO': {
        'def': 'https://www.exomol.com/db/TiO/48Ti-16O/Toto/48Ti-16O__Toto.def',
        'states': 'https://www.exomol.com/db/TiO/48Ti-16O/Toto/48Ti-16O__Toto.states.bz2',
        'trans': 'https://www.exomol.com/db/TiO/48Ti-16O/Toto/48Ti-16O__Toto.trans.bz2',
        'pf': 'https://www.exomol.com/db/TiO/48Ti-16O/Toto/48Ti-16O__Toto.pf'
    },
    'VO': {
        'def': 'https://www.exomol.com/db/VO/51V-16O/HyVO/51V-16O__HyVO.def',
        'states': 'https://www.exomol.com/db/VO/51V-16O/HyVO/51V-16O__HyVO.states.bz2',
        'trans': 'https://www.exomol.com/db/VO/51V-16O/HyVO/', # split into multiple trans files
        'pf': 'https://www.exomol.com/db/VO/51V-16O/HyVO/51V-16O__HyVO.pf'
    },
    'CN': {
        'def': 'https://www.exomol.com/db/CN/12C-14N/KTPSYT/12C-14N__KTPSYT.def',
        'states': 'https://www.exomol.com/db/CN/12C-14N/KTPSYT/12C-14N__KTPSYT.states.bz2',
        'trans': 'https://www.exomol.com/db/CN/12C-14N/KTPSYT/12C-14N__KTPSYT.trans.bz2',
        'pf': 'https://www.exomol.com/db/CN/12C-14N/KTPSYT/12C-14N__KTPSYT.pf'
    },
    'CO': {
        'def': 'https://www.exomol.com/db/CO/12C-16O/Li2015/12C-16O__Li2015.def',
        'states': 'https://www.exomol.com/db/CO/12C-16O/Li2015/12C-16O__Li2015.states.bz2',
        'trans': 'https://www.exomol.com/db/CO/12C-16O/Li2015/12C-16O__Li2015.trans.bz2',
        'pf': 'https://www.exomol.com/db/CO/12C-16O/Li2015/12C-16O__Li2015.pf'
    },
    'H2O': {
        'def': 'https://www.exomol.com/db/H2O/1H2-16O/POKAZATEL/1H2-16O__POKAZATEL.def',
        'states': 'https://www.exomol.com/db/H2O/1H2-16O/POKAZATEL/1H2-16O__POKAZATEL.states.bz2',
        'trans': 'https://www.exomol.com/db/H2O/1H2-16O/POKAZATEL/', # split into multiple trans files 
        'pf': 'https://www.exomol.com/db/H2O/1H2-16O/POKAZATEL/1H2-16O__POKAZATEL.pf'
    }
}

def read_exomolweb_def(molecule):
    def_url = MOLECULE_URLS[molecule]['def']
    resp = requests.get(def_url)
    resp.raise_for_status()
    return resp.text.splitlines()

def get_exomol_states(molecule, dest='./'):
    url = MOLECULE_URLS[molecule]['states']
    file_path = os.path.join(dest, os.path.basename(url))
    resp = requests.get(url)
    resp.raise_for_status()
    with open(file_path, 'wb') as f:
        f.write(resp.content)
    return file_path

def read_exomol_states(molecule, states_file):
    # If file is bz2, decompress to memory
    if states_file.endswith('.bz2'):
        with bz2.BZ2File(states_file) as f:
            text = f.read().decode()
        df = pd.read_csv(StringIO(text), delim_whitespace=True, comment="#", header=None)
    else:
        df = pd.read_csv(states_file, delim_whitespace=True, comment="#", header=None)
    return df


def get_exomol_trans(molecule, E_min, E_max, dest='./'):
    url_base = MOLECULE_URLS[molecule]['trans']
    downloaded_files = []

    if molecule == 'H2O':
        range_min = np.where(H2O_trans_range <= E_min)[0].max()  
        range_max = np.where(H2O_trans_range >= E_max)[0].min()
        splits = H2O_str[range_min:range_max+1]

        for i in range(len(splits) - 1):
            trans_url = f'{url_base}1H2-16O__POKAZATEL__{splits[i]}-{splits[i+1]}.trans.bz2'
            file_path = os.path.join(dest, os.path.basename(trans_url))
            print(f"Downloading {os.path.basename(file_path)} ...")

            # stream download
            with requests.get(trans_url, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                with open(file_path, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            downloaded_files.append(file_path)

    elif molecule == 'VO':
        range_min = np.where(VO_trans_range <= E_min)[0].max()  
        range_max = np.where(VO_trans_range >= E_max)[0].min()
        splits = VO_str[range_min:range_max+1]

        for i in range(len(splits) - 1):
            trans_url = f'{url_base}51V-16O__HyVO__{splits[i]}-{splits[i+1]}.trans.bz2'
            file_path = os.path.join(dest, os.path.basename(trans_url))
            print(f"Downloading {os.path.basename(file_path)} ...")

            with requests.get(trans_url, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                with open(file_path, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            downloaded_files.append(file_path)

    else:
        file_path = os.path.join(dest, os.path.basename(url_base))
        print(f"Downloading {os.path.basename(file_path)} ...")
        with requests.get(url_base, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with open(file_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        downloaded_files.append(file_path)

    print("Download complete.")
    return downloaded_files

def read_exomol_trans(molecule, trans_file):
    if trans_file.endswith('.bz2'):
        with bz2.BZ2File(trans_file) as f:
            text = f.read().decode()
        df = pd.read_csv(StringIO(text), delim_whitespace=True, comment="#", header=None, names=['upper', 'lower', 'A'])
    else:
        df = pd.read_csv(trans_file, delim_whitespace=True, comment="#", header=None, names=['upper', 'lower', 'A'])
    return df

def read_exomol_pf(molecule):
    url = MOLECULE_URLS[molecule]['pf']
    resp = requests.get(url)
    resp.raise_for_status()
    # Example .pf files are generally whitespace separated with two columns: T, Q(T)
    return pd.read_csv(StringIO(resp.text), delim_whitespace=True, comment="#", header=None, names=['T', 'Q'])

def read_exomol_broad(molecule, broadner):
    # TiO, VO, CN, CO, H2O -- broadening file URL needs broadener name
    if molecule == 'TiO':
        broad_url = f'https://www.exomol.com/db/TiO/48Ti-16O/48Ti-16O__{broadner}.broad'
    elif molecule == 'VO':
        broad_url = f'https://www.exomol.com/db/VO/51V-16O/51V-16O__{broadner}.broad'
    elif molecule == 'CN':
        broad_url = f'https://www.exomol.com/db/CN/12C-14N/12C-14N__{broadner}.broad'
    elif molecule == 'CO':
        broad_url = f'https://www.exomol.com/db/CO/12C-16O/12C-16O__{broadner}.broad'
    elif molecule == 'H2O':
        broad_url = f'https://www.exomol.com/db/H2O/H2-16O/H2-16O__{broadner}.broad'
    else:
        raise ValueError('Molecule not recognized. Please choose from TiO, VO, CN, CO, H2O.')
    resp = requests.get(broad_url)
    resp.raise_for_status()
    return resp.text.splitlines()