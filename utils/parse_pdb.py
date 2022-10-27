from typing import List, Dict

def open_pdb(file_path: str, thr_resolution: float = 3.5) -> Dict:
    """
    Read a PDB file and parse it into a dictionary if it meets criteria

    The criteria are:
    - only contains proteins,
    - resolution is known and is not larger than the threshold.

    The output dictionary has the following keys:
    - `'crd_raw'`: a `pandas` (`biopandas`) table with the coordinates,
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