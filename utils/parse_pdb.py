from biopandas.pdb import PandasPdb
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
import numpy as np
from typing import Dict, List


side_chain = {
    "CYS": ["CB", "SG"],
    "ASP": ["CB", "CG", "OD1", "OD2"],
    "SER": ["CB", "OG"],
    "GLN": ["CB", "CG", "CD", "OE1", "NE2"],
    "LYS": ["CB", "CG", "CD", "CE", "NZ"],
    "ILE": ["CB", "CG1", "CG2", "CD1"],
    "PRO": ["CB", "CG", "CD"],
    "THR": ["CB", "OG1", "CG2"],
    "PHE": ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "ASN": ["CB", "CG", "OD1", "ND2"],
    "GLY": [],
    "HIS": ["CB", "CG", "ND1", "CD2", "CE1", "NE2"],
    "LEU": ["CB", "CG", "CD1", "CD2"],
    "ARG": ["CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
    "TRP": ["CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    "ALA": ["CB"],
    "VAL": ["CB", "CG1", "CG2"],
    "GLU": ["CB", "CG", "CD", "OE1", "OE2"],
    "TYR": ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
    "MET": ["CB", "CG", "SD", "CE"],
}

d3to1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
 'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
 'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
 'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

bb_names = ["N", "C", "CA", "O"]

def open_pdb(file_path: str, thr_resolution: float = 3.5) -> Dict:
    """
    Read a PDB file and parse it into a dictionary if it meets criteria

    The criteria are:
    - only contains proteins,
    - resolution is known and is not larger than the threshold.

    The output dictionary has the following keys:
    - `'crd_raw'`: a `pandas` table with the coordinates (the output of `ppdb.df['ATOM']`),
    - `'fasta'`: a dictionary where keys are chain ids and values are fasta sequences.

    Parameters
    ----------
    file_path : str
        the path to the .pdb file
    thr_resolution : float, default 3.5
        the resolution threshold
    
    Output
    ------
    pdb_dict : Dict | None
        the parsed dictionary or `None`, if the criteria are not met

    """

def align_pdb(pdb_dict: Dict, min_length: int = 30, max_length: int = None, max_missing: float = 0.1) -> Dict:
    """
    Align and filter a PDB dictionary

    The filtering criteria are:
    - only contains natural amino acids,
    - number of non-missing residues per chain is not smaller than `min_length`,
    - fraction of missing residues per chain is not larger than `max_missing`,
    - number of residues per chain is not larger than `max_length` (if provided).

    The output is a nested dictionary where first-level keys are chain Ids and second-level keys are the following:
    - `'crd_bb'`: a `numpy` array of shape `(L, 4, 3)` with backbone atom coordinates (N, Ca, C, O),
    - `'crd_sc'`: a `numpy` array of shape `(L, 10, 3)` with sidechain atom coordinates (in a fixed order),
    - `'msk'`: a string of `'+'` and `'-'` of length `L` where `'-'` corresponds to missing residues,
    - `'seq'`: a string of length `L` of residue types,

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
    crd = pdb_dict["crd_raw"]
    fasta = pdb_dict["fasta"]
    output = {}
    if not crd["residue_name"].isin(d3to1.keys()).all():
        return None
    for chain in crd["chain_id"].unique():
        output[chain] = {}
        chain_crd = crd[crd["chain_id"] == chain].reset_index()
        indices = np.unique(chain_crd["residue_number"], return_index=True)[1]
        pdb_seq = "".join([d3to1[x] for x in chain_crd.loc[indices]["residue_name"]])
        aligned_seq = pairwise2.align.globalms(pdb_seq, fasta[chain], 2, -4, -.5, -.1)[0][0]
        output[chain]["seq"] = aligned_seq
        output[chain]["msk"] = (np.array(list(aligned_seq)) != "-").astype(int)
        l = sum(output[chain]["msk"])
        if l < min_length or l / len(aligned_seq) < 1 - max_missing:
            return None
        if max_length is not None and len(aligned_seq) > max_length:
            return None
        crd_arr = np.zeros((len(aligned_seq), 14, 3))
        seq_pos = -1
        pdb_pos = -1
        for i, row in chain_crd.iterrows():
            res_num = row["residue_number"]
            res_name = row["residue_name"]
            atom = row["atom_name"]
            if res_num != pdb_pos:
                seq_pos += 1
                pdb_pos = res_num
                while aligned_seq[seq_pos] == "-":
                    seq_pos += 1
                if d3to1[res_name] != aligned_seq[seq_pos]:
                    print('error 2')
            if atom not in bb_names + side_chain[res_name]:
                if atom in ["OXT", "HXT"]:
                    continue
                return None
            else:
                crd_arr[seq_pos, (bb_names + side_chain[res_name]).index(atom), :] = row[["x_coord", "y_coord", "z_coord"]]
        output[chain]["crd_bb"] = crd_arr[:, : 4, :]
        output[chain]["crd_sc"] = crd_arr[:, 4:, :]
    return output
                
        