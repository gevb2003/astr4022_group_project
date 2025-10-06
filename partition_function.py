# This script reads the .states, .trans, .pf and .broad files from exomol datatbase for a given molecule (TiO, VO, CN, CO, H2O)
import os
import requests
import pandas as pd
from io import StringIO
from io import BytesIO
import bz2
import numpy as np
import pyarrow.csv as pv
import pyarrow.parquet as pq


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

def read_exomolweb_def(molecule, dest='./'):
    # url of file to be downloaded
    def_url = MOLECULE_URLS[molecule]['def']
    # file path of location of file once downloaded
    file_path = os.path.join(dest, os.path.basename(def_url))

    #check if file already exists
    if not os.path.exists(file_path):
        print(f"Downloading {file_path} ...")
        # download in chunks
        with requests.get(def_url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with open(file_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        print(f"Download complete: {file_path}")
    else:
        print(f"File already exists: {file_path}")
    # Read the file 
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().splitlines()

def get_exomol_states(molecule, dest='./'):
    # url of file to be downloaded
    url = MOLECULE_URLS[molecule]['states']
    # file path of once downloaded
    file_path = os.path.join(dest, os.path.basename(url))

    #check if file already exists
    if not os.path.exists(file_path):
        print(f"Downloading {file_path} ...")
        # download in chunks
        with requests.get(url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with open(file_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        print(f"Download complete: {file_path}")
    else:
        print(f"File already exists: {file_path}")

    return file_path

def read_exomol_states(states_file):
    opener = bz2.open if states_file.endswith(".bz2") else open
    with opener(states_file, "rt") as f:  # 't' for text mode
        df = pd.read_csv(f, delim_whitespace=True, comment="#", header=None, engine="c")
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

            # Check if the file is already downloaded 
            if not os.path.exists(file_path):
                print(f"Downloading {os.path.basename(file_path)} ...")

                # stream download
                with requests.get(trans_url, stream=True, timeout=120) as resp:
                    resp.raise_for_status()
                    with open(file_path, 'wb') as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                print(f"Download complete: {file_path}")
            else: 
                print(f"File already exists: {file_path}")
            
            downloaded_files.append(file_path)

    elif molecule == 'VO':
        range_min = np.where(VO_trans_range <= E_min)[0].max()  
        range_max = np.where(VO_trans_range >= E_max)[0].min()
        splits = VO_str[range_min:range_max+1]

        for i in range(len(splits) - 1):
            trans_url = f'{url_base}51V-16O__HyVO__{splits[i]}-{splits[i+1]}.trans.bz2'
            file_path = os.path.join(dest, os.path.basename(trans_url))

            # Check if the file is already downloaded 
            if not os.path.exists(file_path):
                print(f"Downloading {os.path.basename(file_path)} ...")

                with requests.get(trans_url, stream=True, timeout=120) as resp:
                    resp.raise_for_status()
                    with open(file_path, 'wb') as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                print(f"Download complete: {file_path}")
            else: 
                print(f"File already exists: {file_path}")

            downloaded_files.append(file_path)

    else:
        file_path = os.path.join(dest, os.path.basename(url_base))

        # Check if the file is already downloaded 
        if not os.path.exists(file_path):
            print(f"Downloading {os.path.basename(file_path)} ...")
            with requests.get(url_base, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                with open(file_path, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            print(f"Download complete: {file_path}")
        else: 
            print(f"File already exists: {file_path}")
        downloaded_files.append(file_path)

    print("Download complete.")
    return downloaded_files

def read_exomol_trans(trans_file):
    print("Reading trans file")

    opener = bz2.open if trans_file.endswith(".bz2") else open
    with opener(trans_file, "rt") as f:
        # Split on *any amount* of whitespace
        df = pd.read_csv(
            f,
            sep=r"\s+",
            names=["upper", "lower", "A"],
            engine="c",
        )

    print("Finished reading trans file")
    return df

def read_exomol_pf(molecule, dest='./'):
    url = MOLECULE_URLS[molecule]['pf']

    #define where file will be saved
    file_path = os.path.join(dest, os.path.basename(url))

    # Check is file is already downloaded
    if not os.path.exists(file_path):
        # download file in chunks 
        print(f"Downloading {os.path.basename(file_path)} ...")
        with requests.get(url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with open(file_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
    else:
        print(f"File already exists: {file_path}")
    # Create data frame from pf file
    with open(file_path, 'r') as f:
        pf_table = pd.read_csv(f, sep=r'\s+', comment="#", header=None, names=['T', 'Q'])

    return pf_table

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
        raise ValueError('Molecule not recognised. Please choose from TiO, VO, CN, CO, H2O.')
    resp = requests.get(broad_url)
    resp.raise_for_status()
    return resp.text.splitlines()