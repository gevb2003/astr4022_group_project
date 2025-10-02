# This script reads the .states, .trans, .pf and .broad files from exomol datatbase for a given molecule (TiO, VO, CN, CO, H2O)
import os
import requests
import pandas as pd
from io import StringIO

import os
import requests
import pandas as pd
from io import BytesIO
import bz2

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
        'trans': 'https://www.exomol.com/db/VO/51V-16O/HyVO/51V-16O__HyVO.trans.bz2',
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
        'trans': 'https://www.exomol.com/db/H2O/1H2-16O/POKAZATEL/1H2-16O__POKAZATEL.trans.bz2',
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

def get_exomol_trans(molecule, dest='./'):
    url = MOLECULE_URLS[molecule]['trans']
    file_path = os.path.join(dest, os.path.basename(url))
    resp = requests.get(url)
    resp.raise_for_status()
    with open(file_path, 'wb') as f:
        f.write(resp.content)
    return file_path

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