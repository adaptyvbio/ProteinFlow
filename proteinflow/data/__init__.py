import os
import pickle
import urllib
import warnings

import numpy as np
import requests
from Bio import pairwise2
from biopandas.pdb import PandasPdb
from methodtools import lru_cache

from proteinflow.constants import (
    _PMAP,
    ALPHABET,
    ALPHABET_REVERSE,
    ATOM_MASKS,
    BACKBONE_ORDER,
    CDR_ALPHABET,
    CDR_REVERSE,
    CDR_VALUES,
    MAIN_ATOM_DICT,
    SIDECHAIN_ORDER,
)
from proteinflow.data.utils import *


def _download_file(url, local_path):
    """Download a file from a URL to a local path"""
    response = requests.get(url)
    open(local_path, "wb").write(response.content)


def download_pdb(pdb_id, local_folder=".", sabdab=False):
    """
    Download a PDB file from the RCSB PDB database.

    Parameters
    ----------
    pdb_id : str
        PDB ID of the protein to download, can include a biounit index separated
        by a dash (e.g. "1a0a", "1a0a-1")
    local_folder : str, default "."
        Folder to save the downloaded file to
    sabdab : bool, default False
        If True, download from the SAbDab database (Chothia style) instead of RCSB PDB

    Returns
    -------
    local_path : str
        Path to the downloaded file

    """
    if sabdab:
        try:
            url = f"https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/pdb/{pdb_id}/?scheme=chothia"
            local_path = os.path.join(local_folder, f"{pdb_id}.pdb")
            _download_file(url, local_path)
            return local_path
        except:
            raise RuntimeError(f"Could not download {pdb_id}")
    if "-" in pdb_id:
        pdb_id, biounit = pdb_id.split("-")
        filenames = {
            "cif": f"{pdb_id}-assembly{biounit}.cif.gz",
            "pdb": f"{pdb_id}.pdb{biounit}.gz",
        }
        local_name = f"{pdb_id}-{biounit}"
    else:
        filenames = {
            "cif": f"{pdb_id}.cif.gz",
            "pdb": f"{pdb_id}.pdb.gz",
        }
        local_name = pdb_id
    for t in filenames:
        local_path = os.path.join(local_folder, local_name + f".{t}.gz")
        try:
            url = f"https://files.rcsb.org/download/{filenames[t]}"
            _download_file(url, local_path)
            return local_path
        except BaseException:
            pass
    raise RuntimeError(f"Could not download {pdb_id}")


def download_fasta(pdb_id, local_folder="."):
    """
    Download a FASTA file from the RCSB PDB database.

    Parameters
    ----------
    pdb_id : str
        PDB ID of the protein to download
    local_folder : str, default "."
        Folder to save the downloaded file to

    Returns
    -------
    local_path : str
        Path to the downloaded file

    """
    if "-" in pdb_id:
        pdb_id = pdb_id.split("-")[0]
    downloadurl = "https://www.rcsb.org/fasta/entry/"
    pdbfn = pdb_id + "/download"
    local_path = os.path.join(local_folder, f"{pdb_id.lower()}.fasta")

    url = downloadurl + pdbfn
    urllib.request.urlretrieve(url, local_path)
    return local_path


class ProteinEntry:
    def __init__(self, seq, crd, mask, cdr=None):
        """
        Parameters
        ----------
        seq : str
            Amino acid sequence of the protein (one-letter code)
        crd : np.ndarray
            Coordinates of the protein, `'numpy'` array of shape `(L, 4, 3)`,
            in the order of `N, C, CA, O`
        mask : np.ndarray
            Mask array where 1 indicates residues with known coordinates and 0
            indicates missing values
        cdr : np.ndarray
            A `'numpy'` array of shape `(L,)` where CDR residues are marked with the corresponding type (`'H1'`, `'L1'`, ...)
            and non-CDR residues are marked with `'-'`

        """
        self.seq = seq
        self.crd = crd
        self.mask = mask
        self.cdr = cdr

    def sequence(self, encode=False, cdr=None):
        """Get the amino acid sequence of the protein

        Parameters
        ----------
        encode : bool, default False
            If `True`, the sequence is encoded as a `'numpy'` array of integers
            where each integer corresponds to the index of the amino acid in
            `proteinflow.constants.ALPHABET`
        cdr : {"H1", "H2", "H3", "L1", "L2", "L3"}, optional
            If specified, only the CDR region of the specified type is returned

        Returns
        -------
        seq : str or np.ndarray
            Amino acid sequence of the protein (one-letter code) or an encoded
            sequence as a `'numpy'` array of integers

        """
        if cdr is not None and self.cdr is None:
            raise ValueError("CDR information not available")
        if cdr is not None:
            assert cdr in CDR_REVERSE, f"CDR must be one of {list(CDR_REVERSE.keys())}"
        if encode:
            seq = np.array([ALPHABET_REVERSE[aa] for aa in self.seq])
        else:
            seq = self.seq
        if cdr is not None:
            if not encode:
                seq = np.array(list(seq))
            seq = seq[self.cdr == cdr]
            if not encode:
                seq = "".join(seq)
        return seq

    def coordinates(self, mask=False, bb_only=False, cdr=None):
        """Get the coordinates of the protein

        Backbone atoms are in the order of `N, C, CA, O`; for the full-atom
        order see `ProteinEntry.atom_order()` (sidechain atoms come after the
        backbone atoms).

        Parameters
        ----------
        mask : bool, default False
            If `True`, the coordinates of missing residues are set to `np.nan`
        bb_only : bool, default False
            If `True`, only the backbone atoms are returned
        cdr : {"H1", "H2", "H3", "L1", "L2", "L3"}, optional
            If specified, only the CDR region of the specified type is returned

        Returns
        -------
        crd : np.ndarray
            Coordinates of the protein, `'numpy'` array of shape `(L, 14, 3)`
            or `(L, 4, 3)` if `bb_only=True`

        """
        if cdr is not None and self.cdr is None:
            raise ValueError("CDR information not available")
        if cdr is not None:
            assert cdr in CDR_REVERSE, f"CDR must be one of {list(CDR_REVERSE.keys())}"
        crd = self.crd.copy()
        if mask:
            crd[~self.mask] = np.nan
        if cdr is not None:
            crd = crd[self.cdr == cdr]
        return crd

    def mask(self, cdr=None):
        """Get the mask of the protein

        Parameters
        ----------
        cdr : {"H1", "H2", "H3", "L1", "L2", "L3"}, optional
            If specified, only the CDR region of the specified type is returned

        Returns
        -------
        mask : np.ndarray
            Mask array where 1 indicates residues with known coordinates and 0
            indicates missing values

        """
        if cdr is not None and self.cdr is None:
            raise ValueError("CDR information not available")
        if cdr is not None:
            assert cdr in CDR_REVERSE, f"CDR must be one of {list(CDR_REVERSE.keys())}"
        mask = self.mask.copy()
        if cdr is not None:
            mask = mask[self.cdr == cdr]
        return mask

    def cdr(self, encode=False):
        """Get the CDR information of the protein

        Parameters
        ----------
        encode : bool, default False
            If `True`, the CDR information is encoded as a `'numpy'` array of
            integers where each integer corresponds to the index of the CDR
            type in `proteinflow.constants.CDR_ALPHABET`

        Returns
        -------
        cdr : np.ndarray or None
            A `'numpy'` array of shape `(L,)` where CDR residues are marked
            with the corresponding type (`'H1'`, `'L1'`, ...) and non-CDR
            residues are marked with `'-'` or an encoded array of integers
            ir `encode=True`; `None` if CDR information is not available

        """
        if self.cdr is None:
            return None
        if encode:
            cdr = np.array([CDR_REVERSE[aa] for aa in self.cdr])
        else:
            cdr = self.cdr
        return cdr

    def atom_mask(self, cdr=None):
        """Get the atom mask of the protein

        Parameters
        ----------
        cdr : {"H1", "H2", "H3", "L1", "L2", "L3"}, optional
            If specified, only the CDR region of the specified type is returned

        Returns
        -------
        atom_mask : np.ndarray
            Atom mask array where 1 indicates atoms with known coordinates and 0
            indicates missing or non-existing values, shaped `(L, 14, 3)`

        """
        if cdr is not None and self.cdr is None:
            raise ValueError("CDR information not available")
        if cdr is not None:
            assert cdr in CDR_REVERSE, f"CDR must be one of {list(CDR_REVERSE.keys())}"
        atom_mask = np.concatenate([ATOM_MASKS[aa] for aa in self.seq])
        atom_mask[self.mask == 0] = 0
        if cdr is not None:
            atom_mask = atom_mask[self.cdr == cdr]
        return atom_mask

    @staticmethod
    def decode_cdr(cdr):
        """Decode the CDR information

        Parameters
        ----------
        cdr : np.ndarray
            A `'numpy'` array of shape `(L,)` encoded as integers where each
            integer corresponds to the index of the CDR type in
            `proteinflow.constants.CDR_ALPHABET`

        Returns
        -------
        cdr : np.ndarray
            A `'numpy'` array of shape `(L,)` where CDR residues are marked
            with the corresponding type (`'H1'`, `'L1'`, ...) and non-CDR
            residues are marked with `'-'`

        """
        return np.array([CDR_ALPHABET[x] for x in cdr])

    @staticmethod
    def decode_sequence(seq):
        """Decode the amino acid sequence

        Parameters
        ----------
        seq : np.ndarray
            A `'numpy'` array of integers where each integer corresponds to the
            index of the amino acid in `proteinflow.constants.ALPHABET`

        Returns
        -------
        seq : str
            Amino acid sequence of the protein (one-letter code)

        """
        return "".join([ALPHABET[x] for x in seq])

    @staticmethod
    def atom_order():
        """Get the order of atoms in the full-atom representation

        Returns
        -------
        atom_order : dict
            A dictionary where the keys are the one-letter amino acid codes and
            the values are the order of atoms in the full-atom representation

        """
        atom_order = {k: BACKBONE_ORDER + v for k, v in SIDECHAIN_ORDER.items()}
        return atom_order

    @staticmethod
    def from_pdb(path):
        ...

    @staticmethod
    def from_pickle(path):
        """Load a protein entry from a pickle file

        Parameters
        ----------
        path : str
            Path to the pickle file

        Returns
        -------
        entry : ProteinEntry
            A `ProteinEntry` object

        """
        with open(path, "rb") as f:
            data = pickle.load(f)
        seq = data["seq"]
        crd = np.concatenate([data["crd_bb"], data["crd_sc"]], axis=1)
        mask = data["msk"]
        cdr = data.get("cdr", None)
        return ProteinEntry(seq, crd, mask, cdr)

    def to_pdb(self, path):
        ...

    def to_pickle(self, path):
        """Save a protein entry to a pickle file

        The output files are pickled nested dictionaries where first-level keys are chain Ids and second-level keys are the following:
        - `'crd_bb'`: a `numpy` array of shape `(L, 4, 3)` with backbone atom coordinates (N, C, CA, O),
        - `'crd_sc'`: a `numpy` array of shape `(L, 10, 3)` with sidechain atom coordinates (check `proteinflow.sidechain_order()` for the order of atoms),
        - `'msk'`: a `numpy` array of shape `(L,)` where ones correspond to residues with known coordinates and
            zeros to missing values,
        - `'seq'`: a string of length `L` with residue types.

        In a SAbDab datasets, an additional key is added to the dictionary:
        - `'cdr'`: a `'numpy'` array of shape `(L,)` where CDR residues are marked with the corresponding type (`'H1'`, `'L1'`, ...)
            and non-CDR residues are marked with `'-'`.

        Parameters
        ----------
        path : str
            Path to the pickle file

        """
        data = {
            "seq": self.seq,
            "crd_bb": self.crd[:, :4],
            "crd_sc": self.crd[:, 4:],
            "msk": self.mask,
        }
        if self.cdr is not None:
            data["cdr"] = self.cdr
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def dihedral_angles(self):
        """Calculate the backbone dihedral angles (phi, psi) of the protein

        Returns
        -------
        angles : np.ndarray
            A `'numpy'` array of shape `(L, 2)` with backbone dihedral angles
            (phi, psi) in degrees; missing values are marked with zeros

        """
        angles = []
        # N, C, Ca, O
        # psi
        p = self.crd[:-1, [0, 2, 1], :]
        p = np.concatenate([p, self.crd[1:, [0], :]], 1)
        p = np.pad(p, ((0, 1), (0, 0), (0, 0)))
        angles.append(_dihedral_angle(p, self.mask))
        # phi
        p = self.crd[:-1, [1], :]
        p = np.concatenate([p, self.crd[1:, [0, 2, 1]]], 1)
        p = np.pad(p, ((1, 0), (0, 0), (0, 0)))
        angles.append(_dihedral_angle(p, self.mask))
        angles = np.stack(angles, -1)
        return angles

    def secondary_structure(self):
        """Calculate the secondary structure of the protein

        Returns
        -------
        sse : np.ndarray
            A `'numpy'` array of shape `(L, 3)` with secondary structure
            elements encoded as one-hot vectors (alpha-helix, beta-sheet, loop);
            missing values are marked with zeros

        """
        sse_map = {"c": [0, 0, 1], "b": [0, 1, 0], "a": [1, 0, 0], "": [0, 0, 0]}
        sse = _annotate_sse(self.crd[:, :4])
        sse = np.array([sse_map[x] for x in sse]) * self.mask[:, None]
        return sse

    def sidechain_coordinates(self):
        """Get the sidechain coordinates of the protein

        Returns
        -------
        crd : np.ndarray
            A `'numpy'` array of shape `(L, 10, 3)` with sidechain atom
            coordinates (check `proteinflow.sidechain_order()` for the order of
            atoms); missing values are marked with zeros

        """
        return self.crd[:, 4:] * self.mask[:, None, None]

    def chemical_features(self):
        """Calculate chemical features of the protein

        Returns
        -------
        features : np.ndarray
            A `'numpy'` array of shape `(L, 4)` with chemical features of the
            protein (hydropathy, volume, charge, polarity, acceptor/donor); missing
            values are marked with zeros

        """
        features = np.array([_PMAP(x) for x in self.seq])
        return features

    def sidechain_orientation(self):
        """Calculate the (global) sidechain orientation of the protein

        Returns
        -------
        orientation : np.ndarray
            A `'numpy'` array of shape `(L, 3)` with sidechain orientation

        """
        crd_bb = self.crd[:, :4]
        crd_sc = self.crd[:, 4:]
        seq = self.sequence(encode=True)
        orientation = np.zeros((crd_sc.shape[0], 3))
        for i in range(1, 21):
            if MAIN_ATOM_DICT[i] is not None:
                orientation[seq == i] = (
                    crd_sc[seq == i, MAIN_ATOM_DICT[i], :] - crd_bb[seq == i, 2, :]
                )
            else:
                S_mask = self.seq == i
                orientation[S_mask] = np.random.rand(*orientation[S_mask].shape)
        orientation /= np.expand_dims(np.linalg.norm(orientation, axis=-1), -1) + 1e-7
        return orientation


class PDBEntry:
    def __init__(self, pdb_path, fasta_path):
        """A class for parsing PDB files

        Parameters
        ----------
        pdb_path : str
            Path to the PDB file
        fasta_path : str
            Path to the FASTA file

        """
        self.pdb_path = pdb_path
        self.fasta_path = fasta_path
        self.pdb_id = os.path.basename(self.fasta_path).split(".")[0]
        self.crd_df, self.seq_df, self.fasta_dict = self._parse()

    @staticmethod
    def from_id(pdb_id, local_folder="."):
        """Initialize a `PDBEntry` object from a PDB Id

        Downloads the PDB and FASTA files to the local folder.

        Parameters
        ----------
        pdb_id : str
            PDB Id of the protein
        local_folder : str, default '.'
            Folder where the downloaded files will be stored

        Returns
        -------
        entry : PDBEntry
            A `PDBEntry` object

        """
        pdb_path = download_pdb(pdb_id, local_folder)
        fasta_path = download_fasta(pdb_id, local_folder)
        return PDBEntry(pdb_path=pdb_path, fasta_path=fasta_path)

    @staticmethod
    def parse_fasta(fasta_path):
        """Read a fasta file

        Parameters
        ----------
        fasta_path : str
            Path to the fasta file

        Returns
        -------
        out_dict : dict
            A dictionary containing all the (author) chains in a fasta file (keys)
            and their corresponding sequence (values)

        """
        with open(fasta_path) as f:
            lines = np.array(f.readlines())

        indexes = np.array([k for k, l in enumerate(lines) if l[0] == ">"])
        starts = indexes + 1
        ends = list(indexes[1:]) + [len(lines)]
        names = lines[indexes]
        seqs = ["".join(lines[s:e]).replace("\n", "") for s, e in zip(starts, ends)]

        out_dict = {}
        for name, seq in zip(names, seqs):
            for chain in _retrieve_chain_names(name):
                out_dict[chain] = seq

        return out_dict

    def _parse(self, chains=None):
        cif = self.pdb_path.endswith("cif.gz")

        # download fasta and check if it contains only proteins
        try:
            seqs_dict = self.parse_fasta(self.fasta_path)
        except FileNotFoundError:
            raise PDBError("Fasta file not found")

        # load coordinates in a nice format
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if cif:
                    p = CustomMmcif().read_mmcif(self.pdb_path).get_model(1)
                else:
                    p = PandasPdb().read_pdb(self.pdb_path).get_model(1)
        except FileNotFoundError:
            raise PDBError("PDB / mmCIF file downloaded but not found")
        crd_df = p.df["ATOM"]
        seq_df = p.amino3to1()

        # retrieve sequences that are relevant for this PDB from the fasta file
        if chains is None:
            chains = p.df["ATOM"]["chain_id"].unique()
        seqs_dict = {k.upper(): v for k, v in seqs_dict.items()}
        if all([len(x) == 3 and len(set(list(x))) == 1 for x in seqs_dict.keys()]):
            seqs_dict = {k[0]: v for k, v in seqs_dict.items()}

        if not {x.split("-")[0].upper() for x in chains}.issubset(
            set(list(seqs_dict.keys()))
        ):
            raise PDBError("Some chains in the PDB do not appear in the fasta file")

        fasta_dict = {k: seqs_dict[k.split("-")[0].upper()] for k in chains}
        return crd_df, seq_df, fasta_dict

    def pdb_df(self, chain=None):
        """Return the PDB dataframe

        If `chain` is provided, only information for this chain is returned.

        Parameters
        ----------
        chain : str, optional
            Chain identifier

        Returns
        -------
        df : pd.DataFrame
            A `BioPandas` style dataframe containing the PDB information

        """
        if chain is None:
            return self.crd_df
        else:
            return self.crd_df[self.crd_df["chain_id"] == chain]

    def sequence_df(self, chain=None):
        """Return the sequence dataframe

        If `chain` is provided, only information for this chain is returned.

        Parameters
        ----------
        chain : str, optional
            Chain identifier

        Returns
        -------
        df : pd.DataFrame
            A dataframe containing the sequence and chain information
            (analogous to the `BioPandas.pdb.PandasPdb.amino3to1` method output)

        """
        if chain is None:
            return self.seq_df
        else:
            return self.seq_df[self.seq_df["chain_id"] == chain]

    def fasta(self):
        """Return the fasta dictionary

        Returns
        -------
        fasta_dict : dict
            A dictionary containing all the (author) chains in a fasta file (keys)
            and their corresponding sequence (values)

        """
        return self.fasta_dict

    def chains(self):
        """Return the chains in the PDB

        Returns
        -------
        chains : list
            A list of chain identifiers

        """
        return list(self.fasta_dict.keys())

    @lru_cache()
    def _pdb_sequence(self, chain):
        """Return the PDB sequence for a given chain ID"""
        return "".join(self.sequence_df(chain)["residue_name"])

    @lru_cache()
    def _align(self, chain):
        """Align the PDB sequence to the FASTA sequence for a given chain ID"""
        chain_crd = self.pdb_df(chain)
        pdb_seq = self._pdb_sequence(chain)
        # aligner = PairwiseAligner()
        # aligner.match_score = 2
        # aligner.mismatch_score = -10
        # aligner.open_gap_score = -0.5
        # aligner.extend_gap_score = -0.1
        # aligned_seq, fasta_seq = aligner.align(pdb_seq, fasta[chain])[0]
        aligned_seq, fasta_seq, *_ = pairwise2.align.globalms(
            pdb_seq, self.fasta_dict[chain], 2, -10, -0.5, -0.1
        )[0]
        if "-" in fasta_seq or "".join([x for x in aligned_seq if x != "-"]) != pdb_seq:
            raise PDBError("Incorrect alignment")
        return aligned_seq, fasta_seq

    def alignment(self, chains=None):
        """Return the alignment between the PDB and the FASTA sequence

        Parameters
        ----------
        chains : list, optional
            A list of chain identifiers (if not provided, all chains are aligned)

        Returns
        -------
        alignment : dict
            A dictionary containing the aligned sequences for each chain

        """
        if chains is None:
            chains = self.chains()
        return {chain: self._align(chain) for chain in chains}


class SAbDabEntry(PDBEntry):
    def __init__(
        self,
        pdb_path,
        fasta_path,
        heavy_chain=None,
        light_chain=None,
        antigen_chains=None,
    ):
        """
        Parameters
        ----------
        pdb_path : str
            Path to the PDB file
        fasta_path : str
            Path to the FASTA file
        heavy_chain : str, optional
            Heavy chain identifier (author chain name)
        light_chain : str, optional
            Light chain identifier (author chain name)
        antigen_chains : list, optional
            List of antigen chain identifiers (author chain names)

        """
        if heavy_chain is None and light_chain is None:
            raise PDBError("At least one chain must be provided")
        self.chain_dict = {
            "heavy": heavy_chain,
            "light": light_chain,
        }
        if antigen_chains is None:
            antigen_chains = []
        self.chain_dict["antigen"] = antigen_chains
        super().__init__(pdb_path, fasta_path)

    @staticmethod
    def from_id(
        pdb_id,
        local_folder=".",
        light_chain=None,
        heavy_chain=None,
        antigen_chains=None,
    ):
        pdb_path = download_pdb(pdb_id, local_folder)
        fasta_path = download_fasta(pdb_id, local_folder)
        return SAbDabEntry(
            pdb_path=pdb_path,
            fasta_path=fasta_path,
            light_chain=light_chain,
            heavy_chain=heavy_chain,
            antigen_chains=antigen_chains,
        )

    def heavy_chain(self):
        """Return the heavy chain identifier

        Returns
        -------
        chain : str
            The heavy chain identifier

        """
        return self.chain_dict["heavy"]

    def light_chain(self):
        """Return the light chain identifier

        Returns
        -------
        chain : str
            The light chain identifier

        """
        return self.chain_dict["light"]

    def antigen_chains(self):
        """Return the antigen chain identifiers

        Returns
        -------
        chains : list
            The antigen chain identifiers

        """
        return self.chain_dict["antigen"]

    def chains(self):
        """Return the chains in the PDB

        Returns
        -------
        chains : list
            A list of chain identifiers

        """
        return [self.heavy_chain(), self.light_chain()] + self.antigen_chains()

    def chain_type(self, chain):
        """Return the type of a chain

        Parameters
        ----------
        chain : str
            Chain identifier

        Returns
        -------
        chain_type : str
            The type of the chain (heavy, light or antigen)

        """
        for k, v in self.chain_dict.items():
            if chain in v:
                return k
        raise PDBError("Chain not found")

    @lru_cache()
    def _get_cdr(self, chain, align_to_fasta=False):
        chain_crd = self.pdb_df(chain)
        chain_type = self.chain_type(chain)[0].upper()
        if chain_type not in ["H", "L"]:
            return None
        pdb_seq = self._pdb_sequence(chain)
        if "insertion" in chain_crd.columns:
            chain_crd["residue_number"] = chain_crd.apply(
                lambda row: f"{row['residue_number']}_{row['insertion']}", axis=1
            )
        unique_numbers = chain_crd["residue_number"].unique()
        if len(unique_numbers) != len(pdb_seq):
            raise PDBError("Inconsistencies in the biopandas dataframe")
        cdr_arr = [CDR_VALUES[chain_type][int(x.split("_")[0])] for x in unique_numbers]
        cdr_arr = np.array(cdr_arr)
        if align_to_fasta:
            aligned_seq, fasta_seq = self._align(chain)
            aligned_seq_arr = np.array(list(aligned_seq))
        cdr_arr[aligned_seq_arr != "-"] = _get_chothia_cdr(unique_numbers, chain_type)

    def cdr(self):
        if "insertion" in chain_crd.columns:
            chain_crd["residue_number"] = chain_crd.apply(
                lambda row: f"{row['residue_number']}_{row['insertion']}", axis=1
            )
        unique_numbers = chain_crd["residue_number"].unique()
        if len(unique_numbers) != len(pdb_seq):
            raise PDBError("Inconsistencies in the biopandas dataframe")
        arr = [CDR_VALUES[chain_type][int(x.split("_")[0])] for x in num_array]
        return np.array(arr)
