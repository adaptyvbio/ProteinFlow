from typing import List, Dict
import subprocess
import gzip
import shutil
import urllib.request
import os
import sys
import re
import numpy as np
import pickle as pkl
from Bio.Seq import Seq
from Bio.PDB import PDBParser
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.PDBIO import Select
from Bio.PDB.parse_pdb_header import parse_pdb_header
from biopandas.pdb import PandasPdb




class PDBError(ValueError):
    pass


class SelectHeavyAtoms(Select):

    def accept_residue(self, residue):
        return residue.id[0] == ' '

    def accet_atom(self, atom):
        return atom.id[0] != 'H'


def download_fasta(pdbcode, datadir):
    """
    Downloads a fasta file from the Internet and saves it in a data directory.
    For informations about the download url, cf `https://www.rcsb.org/pages/download/http#structures`
    :param pdbcode: The standard PDB ID e.g. '3ICB' or '3icb'
    :param datadir: The directory where the downloaded file will be saved
    :return: the full path to the downloaded PDB file or None if something went wrong
    """

    downloadurl = "https://www.rcsb.org/fasta/entry/"
    pdbfn = pdbcode + "/download"
    outfnm = os.path.join(datadir, pdbcode + '.fasta')
    
    url = downloadurl + pdbfn
    try:
        urllib.request.urlretrieve(url, outfnm)
        return outfnm
    
    except Exception as err:
        #print(str(err), file=sys.stderr)
        return None, err


def validate(seq, alphabet='dna'):

    """
    Check that a given sequence contains only proteic residues
    """
    
    alphabets = {'dna': re.compile('^[acgtn]*$', re.I), 
             'protein': re.compile('^[acdefghiklmnpqrstvwy]*$', re.I)}

    return alphabets[alphabet].search(seq) is not None


def detect_non_proteins(fasta_file):

    """
    Detect if a fasta contains residues that do not belong to a protein (DNA, RNA, non-canonical amino acids, ...)
    """

    with open(fasta_file, 'r') as f:

        seq = Seq(''.join([line.replace('\n', '') for line in f.readlines() if line[0] != '>']))
    
    return validate(str(seq), 'dna') or not validate(str(seq), 'protein')


def retrieve_author_chain(chain):

    """
    Retrieve the (author) chain names present in the chain section (delimited by '|' chars) of a header line in a fasta file
    """

    if 'auth' in chain:
        return chain.split(' ')[-1][ : -1]
    
    return chain


def retrieve_chain_names(entry):

    """
    Retrieve the (author) chain names present in one header line of a fasta file (line that begins with '>')
    """

    entry = entry.split('|')[1]

    if 'Chains' in entry:
        return [retrieve_author_chain(e) for e in entry[7 : ].split(', ')]
    
    return [retrieve_author_chain(entry[6 : ])]


def retrieve_fasta_chains(fasta_file):

    """
    Return a dictionary containing all the (author) chains in a fasta file (keys) and their corresponding sequence
    """

    with open(fasta_file, 'r') as f:
        lines = np.array(f.readlines())
    
    indexes = np.array([k for k, l in enumerate(lines) if l[0] == '>'])
    starts = indexes + 1
    ends = list(indexes[1 : ]) + [len(lines)]
    names = lines[indexes]
    seqs = [''.join(lines[s : e]).replace('\n', '') for s, e in zip(starts, ends)]

    out_dict = {}
    for name, seq in zip(names, seqs):
        for chain in retrieve_chain_names(name):
            out_dict[chain] = seq
    
    return out_dict


def retrieve_pdb_resolution(pdb_id):

    """
    Find the resolution of the PDB by downloading the PDB from the web
    """

    pdb_file = 'pdb' + pdb_id + '.ent.gz'
    pdb_unzipped = pdb_id + '_full' + '.pdb'
    download_path = 's3://pdbsnapshots/20220103/pub/pdb/data/structures/all/pdb/' + pdb_file
    subprocess.run(['aws', 's3', 'cp', '--no-sign-request', download_path, '.'])

    with gzip.open(pdb_file, 'rb') as f_in:
        with open(pdb_unzipped, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    subprocess.run(['rm', pdb_file])
    header = parse_pdb_header(pdb_unzipped)
    subprocess.run(['rm', pdb_unzipped])
    return header['resolution']


def check_resolution(pdb_id, resolution_dict):

    """
    Find the resolution of the PDB by first checking into the resolution dictionary and then by downloading the PDB from the web if necessary
    """

    with open(resolution_dict, 'rb') as f:
        res_dict = pkl.load(f)
    
    if pdb_id in res_dict.keys():
        resolution = res_dict[pdb_id]
    
    else:
        resolution = retrieve_pdb_resolution(pdb_id)
        res_dict[pdb_id] = resolution
        with open(resolution_dict, 'wb') as f:
            pkl.dump(res_dict, f)
    
    return resolution


def open_pdb(file_path: str, resolution_dict: str, thr_resolution: float = 3.5) -> Dict:
    """
    Read a PDB file and parse it into a dictionary if it meets criteria

    The criteria are:
    - only contains proteins,
    - resolution is known and is not larger than the threshold.

    The output dictionary has the following keys:
    - 'pdb_id': a 4 letters string corresponding to the PDB id
    - 'biounit': a positive integer indexing the biounit wrt its PDB
    - `'crd_raw'`: a `pandas` table with the coordinates (the output of `ppdb.df['ATOM']`),
    - `'fasta'`: a dictionary where keys are chain ids and values are fasta sequences.

    Parameters
    ----------
    file_path : str
        the path to the .pdb{i}.gz file (i is a positive integer)
    resolution_dict : str
        the path to the pkl file containing a dictionary that stores the resolutions of the PDBs
    thr_resolution : float, default 3.5
        the resolution threshold
    
    Output
    ------
    pdb_dict : Dict | None
        the parsed dictionary or `None`, if the criteria are not met

    """

    pdb = file_path.split('.')
    biounit = int(pdb[1][3 : ])
    pdb = pdb[0]
    pdb_path = pdb + '.pdb'
    out_dict = {'pdb_id' : pdb,
                'biounit' : biounit}
    
    # check that the resolution is sufficient
    resolution = check_resolution(pdb, resolution_dict)
    if resolution is None:
        raise PDBError("Resolution was not indicated.")
    
    if resolution > thr_resolution:
        raise PDBError(f"Resolution is > {thr_resolution}")

    # unzip PDB
    with gzip.open(file_path, 'rb') as f_in:
        with open(pdb_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    subprocess.run(['rm', file_path])

    # download fasta and check if it contains only proteins
    fasta_path = download_fasta(pdb, '.')

    if fasta_path[0] == None:
        raise PDBError("Problems downloading fasta file.\n" + str(fasta_path[1]))
    
    if detect_non_proteins(fasta_path):
        raise PDBError("The PDB contains non proteic residues (DNA, RNA, non-canonical amino acids, ...)")

    # load PDB and save it back with all hydrogens and hetero-atoms removed
    p = PDBParser(QUIET=True)
    model = p.get_structure('test', pdb_path)[0]
    io = PDBIO()
    io.set_structure(model)
    io.save(pdb_path, SelectHeavyAtoms())

    # load coordinates in a nice format
    p = PandasPdb().read_pdb(pdb_path).df['ATOM']
    subprocess.run(['rm', pdb_path])
    out_dict['crd_raw'] = p

    # retrieve sequences that are relevant for this PDB from the fasta file
    seqs_dict = retrieve_fasta_chains(fasta_path)
    subprocess.run(['rm', fasta_path])
    chains = np.unique(p['chain_id'].values)

    if not set(chains).issubset(set(list(seqs_dict.keys()))):
        raise PDBError("Some chains in the PDB do not appear in the fasta file.")
    
    out_dict['fasta'] = {k : seqs_dict[k] for k in chains}

    return out_dict


def align_pdb(pdb_dict: Dict, min_length: int = 30, max_length: int = None, max_missing: float = 0.1) -> Dict:
    """
    Align and filter a PDB dictionary

    The filtering criteria are:
    - only contains natural amino acids,
    - number of non-missing residues per chain is not smaller than `min_length`,
    - fraction of missing residues per chain is not larger than `max_missing`,
    - number of residues per chain is not larger than `max_length` (if provided).

    The output dictionary has the following keys:
    - `'crd_bb'`: a `numpy` array of shape `(L, 4, 3)` with backbone atom coordinates (N, Ca, C, O),
    - `'crd_sc'`: a `numpy` array of shape `(L, 10, 3)` with sidechain atom coordinates (in a fixed order),
    - `'msk'`: a string of `'+'` and `'-'` of length `L` where `'-'` corresponds to missing residues,
    - `'seq'`: a string of length `L` of residue types.

    Parameters
    ----------
    pdb_dict : Dict
        the output of `open_pdb`
    min_length : int, default 30
        the minimum number of non-missing residues per chain
    max_length : int, optional
        the maximum number of residues per chain


    Returns
    -------
    pdb_dict : Dict | None
        the parsed dictionary or `None`, if the criteria are not met
        
    """