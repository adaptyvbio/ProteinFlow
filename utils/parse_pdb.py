from typing import List, Dict


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
    output = {}
    if not crd["residue_name"].isin(d3to1.keys()).all():
        print('error')
    for chain in crd["chain_id"].unique():
        chain_crd = crd[crd["chain_id"] == chain]
        indices = np.unique(chain_crd["residue_number"], return_index=True)[1]
        pdb_seq = "".join([d3to1[x] for x in chain_crd.loc[indices]["residue_name"]])
        output["seq"] = pairwise2.align.globalms(pdb_seq, fasta[chain], 2, -1, -.5, -.1)[0][0]
        output["msk"] = (np.array(list(output["seq"])) != "-").astype(int)
        l = sum(output["msk"])
        if l < min_length or l > max_length:
            print('error')
        output["crd_bb"] = np.zeros((l, 4, 3))
        output["crd_sc"] = np.zeros((l, 10, 3))
        bb_names = ["N", "C", "CA", "O"]

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