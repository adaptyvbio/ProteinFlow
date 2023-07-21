"""
Classes for downloading and manipulating protein data.

- `ProteinEntry`: a class for manipulating proteinflow pickle files,
- `PDBEntry`: a class for manipulating raw PDB files,
- `SAbDabEntry`: a class for manipulating SAbDab files with specific methods for antibody data.

A `ProteinEntry` object can be created from a proteinflow pickle file, a PDB file or a SAbDab file directly
and can be used to process the data and extract additional features. The processed data can be saved as a
proteinflow pickle file or a PDB file.

"""
import os
import pickle
import tempfile
import urllib
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import py3Dmol
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
    COLORS,
    D3TO1,
    MAIN_ATOM_DICT,
    SIDECHAIN_ORDER,
)
from proteinflow.data.utils import (
    CustomMmcif,
    PDBBuilder,
    PDBError,
    _annotate_sse,
    _Atom,
    _dihedral_angle,
    _retrieve_chain_names,
)
from proteinflow.download import download_fasta, download_pdb


def interpolate_coords(crd, mask, fill_ends=True):
    """Fill in missing values in a coordinates array with linear interpolation.

    Parameters
    ----------
    crd : np.ndarray
        Coordinates array of shape `(L, 4, 3)`
    mask : np.ndarray
        Mask array of shape `(L,)` where 1 indicates residues with known coordinates and 0
        indicates missing values
    fill_ends : bool, default True
        If `True`, fill in missing values at the ends of the protein sequence with the edge values;
        otherwise fill them in with zeros

    Returns
    -------
    crd : np.ndarray
        Interpolated coordinates array of shape `(L, 4, 3)`
    mask : np.ndarray
        Interpolated mask array of shape `(L,)` where 1 indicates residues with known or interpolated
        coordinates and 0 indicates missing values

    """
    crd[(1 - mask).astype(bool)] = np.nan
    df = pd.DataFrame(crd.reshape((crd.shape[0], -1)))
    crd = df.interpolate(limit_area="inside" if not fill_ends else None).values.reshape(
        crd.shape
    )
    if not fill_ends:
        nan_mask = np.isnan(crd)  # in the middle the nans have been interpolated
        interpolated_mask = np.zeros_like(mask)
        interpolated_mask[~np.isnan(crd[:, 0, 0])] = 1
        crd[nan_mask] = 0
    else:
        interpolated_mask = np.ones_like(crd[:, :, 0])
    return crd, mask


class ProteinEntry:
    """A class to interact with proteinflow data files."""

    ATOM_ORDER = {k: BACKBONE_ORDER + v for k, v in SIDECHAIN_ORDER.items()}
    """A dictionary mapping 3-letter residue names to the order of atoms in the coordinates array."""

    def __init__(self, seqs, crds, masks, chain_ids, cdrs=None):
        """Initialize a `ProteinEntry` object.

        Parameters
        ----------
        seqs : list of str
            Amino acid sequences of the protein (one-letter code)
        crds : list of np.ndarray
            Coordinates of the protein, `'numpy'` arrays of shape `(L, 4, 3)`,
            in the order of `N, C, CA, O`
        masks : list of np.ndarray
            Mask arrays where 1 indicates residues with known coordinates and 0
            indicates missing values
        cdrs : list of np.ndarray
            `'numpy'` arrays of shape `(L,)` where CDR residues are marked with the corresponding type (`'H1'`, `'L1'`, ...)
            and non-CDR residues are marked with `'-'`
        chain_ids : list of str
            Chain IDs of the protein

        """
        self.seq = {x: seq for x, seq in zip(chain_ids, seqs)}
        self.crd = {x: crd for x, crd in zip(chain_ids, crds)}
        self.mask = {x: mask for x, mask in zip(chain_ids, masks)}
        self.mask_original = {x: mask for x, mask in zip(chain_ids, masks)}
        if cdrs is None:
            cdrs = [None for _ in chain_ids]
        self.cdr = {x: cdr for x, cdr in zip(chain_ids, cdrs)}

    def interpolate_coords(self, fill_ends=True):
        """Fill in missing values in the coordinates arrays with linear interpolation.

        Parameters
        ----------
        fill_ends : bool, default True
            If `True`, fill in missing values at the ends of the protein sequence with the edge values;
            otherwise fill them in with zeros

        """
        for chain in self.get_chains():
            self.crd[chain], self.mask[chain] = interpolate_coords(
                self.crd[chain], self.mask[chain], fill_ends=fill_ends
            )

    def cut_missing_edges(self):
        """Cut off the ends of the protein sequence that have missing coordinates."""
        for chain in self.get_chains():
            mask = self.mask[chain]
            known_ind = np.where(mask == 1)[0]
            start, end = known_ind[0], known_ind[-1] + 1
            self.seq[chain] = self.seq[chain][start:end]
            self.crd[chain] = self.crd[chain][start:end]
            self.mask[chain] = self.mask[chain][start:end]
            if self.cdr[chain] is not None:
                self.cdr[chain] = self.cdr[chain][start:end]

    @lru_cache()
    def get_chains(self):
        """Get the chain IDs of the protein.

        Returns
        -------
        chains : list of str
            Chain IDs of the protein

        """
        return sorted(self.seq.keys())

    def _get_chains_list(self, chains):
        """Get a list of chains to iterate over."""
        if chains is None:
            chains = self.get_chains()
        return chains

    def get_chain_type_dict(self, chains=None):
        """Get the chain types of the protein.

        If the CDRs are not annotated, this function will return `None`.

        Parameters
        ----------
        chains : list of str, default None
            Chain IDs to consider

        Returns
        -------
        chain_type_dict : dict
            A dictionary with keys `'heavy'`, `'light'` and `'antigen'` and values
            the corresponding chain IDs

        """
        if not self.has_cdr():
            return None
        chain_type_dict = {"antigen": []}
        chains = self._get_chains_list(chains)
        for chain, cdr in self.cdr.items():
            if chain not in chains:
                continue
            u = np.unique(cdr)
            if "H1" in u:
                chain_type_dict["heavy"] = chain
            elif "L1" in u:
                chain_type_dict["light"] = chain
            else:
                chain_type_dict["antigen"].append(chain)
        return chain_type_dict

    def get_length(self, chains=None):
        """Get the total length of a set of chains.

        Parameters
        ----------
        chain : str, optional
            Chain ID; if `None`, the length of the whole protein is returned

        Returns
        -------
        length : int
            Length of the chain

        """
        chains = self._get_chains_list(chains)
        return sum([len(self.seq[x]) for x in chains])

    def get_cdr_length(self, chains):
        """Get the length of the CDR regions of a set of chains.

        Parameters
        ----------
        chain : str
            Chain ID

        Returns
        -------
        length : int
            Length of the CDR regions of the chain

        """
        if not self.has_cdr():
            return {x: None for x in ["H1", "H2", "H3", "L1", "L2", "L3"]}
        return {
            x: len(self.get_sequence(chains=chains, cdr=x))
            for x in ["H1", "H2", "H3", "L1", "L2", "L3"]
        }

    def has_cdr(self):
        """Check if the protein is from the SAbDab database.

        Returns
        -------
        is_sabdab : bool
            True if the protein is from the SAbDab database

        """
        return list(self.cdr.values())[0] is not None

    def __len__(self):
        """Get the total length of the protein chains."""
        return self.get_length(self.get_chains())

    def get_sequence(self, chains=None, encode=False, cdr=None, only_known=False):
        """Get the amino acid sequence of the protein.

        Parameters
        ----------
        chains : list of str, optional
            If specified, only the sequences of the specified chains is returned (in the same order);
            otherwise, all sequences are concatenated in alphabetical order of the chain IDs
        encode : bool, default False
            If `True`, the sequence is encoded as a `'numpy'` array of integers
            where each integer corresponds to the index of the amino acid in
            `proteinflow.constants.ALPHABET`
        cdr : {"H1", "H2", "H3", "L1", "L2", "L3"}, optional
            If specified, only the CDR region of the specified type is returned
        only_known : bool, default False
            If `True`, only the residues with known coordinates are returned

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
        chains = self._get_chains_list(chains)
        seq = "".join([self.seq[c] for c in chains])
        if encode:
            seq = np.array([ALPHABET_REVERSE[aa] for aa in seq])
        elif cdr is not None or only_known:
            seq = np.array(list(seq))
        if cdr is not None:
            cdr_arr = self.get_cdr(chains=chains)
            seq = seq[cdr_arr == cdr]
        if only_known:
            seq = seq[self.get_mask(chains=chains, cdr=cdr).astype(bool)]
        if not encode and not isinstance(seq, str):
            seq = "".join(seq)
        return seq

    def get_coordinates(self, chains=None, bb_only=False, cdr=None, only_known=False):
        """Get the coordinates of the protein.

        Backbone atoms are in the order of `N, C, CA, O`; for the full-atom
        order see `ProteinEntry.ATOM_ORDER` (sidechain atoms come after the
        backbone atoms).

        Parameters
        ----------
        chains : list of str, optional
            If specified, only the coordinates of the specified chains are returned (in the same order);
            otherwise, all coordinates are concatenated in alphabetical order of the chain IDs
        bb_only : bool, default False
            If `True`, only the backbone atoms are returned
        cdr : {"H1", "H2", "H3", "L1", "L2", "L3"}, optional
            If specified, only the CDR region of the specified type is returned
        only_known : bool, default False
            If `True`, only return the coordinates of residues with known coordinates

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
        chains = self._get_chains_list(chains)
        crd = np.concatenate([self.crd[c] for c in chains], axis=0)
        if cdr is not None:
            crd = crd[self.cdr == cdr]
        if bb_only:
            crd = crd[:, :4, :]
        if only_known:
            crd = crd[self.get_mask(chains=chains, cdr=cdr).astype(bool)]
        return crd

    def get_mask(self, chains=None, cdr=None, original=False):
        """Get the mask of the protein.

        Parameters
        ----------
        chains : list of str, optional
            If specified, only the masks of the specified chains are returned (in the same order);
            otherwise, all masks are concatenated in alphabetical order of the chain IDs
        cdr : {"H1", "H2", "H3", "L1", "L2", "L3"}, optional
            If specified, only the CDR region of the specified type is returned
        original : bool, default False
            If `True`, return the original mask (before interpolation)

        Returns
        -------
        mask : np.ndarray
            Mask array where 1 indicates residues with known coordinates and 0
            indicates missing values
        chains : list of str, optional
            If specified, only the masks of the specified chains are returned (in the same order);
            otherwise, all masks are concatenated in alphabetical order of the chain IDs

        """
        if cdr is not None and self.cdr is None:
            raise ValueError("CDR information not available")
        if cdr is not None:
            assert cdr in CDR_REVERSE, f"CDR must be one of {list(CDR_REVERSE.keys())}"
        chains = self._get_chains_list(chains)
        mask = np.concatenate(
            [self.mask_original[c] if original else self.mask[c] for c in chains],
            axis=0,
        )
        if cdr is not None:
            mask = mask[self.cdr == cdr]
        return mask

    def get_cdr(self, chains=None, encode=False):
        """Get the CDR information of the protein.

        Parameters
        ----------
        chains : list of str, optional
            If specified, only the CDR information of the specified chains is
            returned (in the same order); otherwise, all CDR information is concatenated in
            alphabetical order of the chain IDs
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
        chains : list of str, optional
            If specified, only the CDR information of the specified chains is
            returned (in the same order); otherwise, all CDR information is concatenated in
            alphabetical order of the chain IDs

        """
        chains = self._get_chains_list(chains)
        if self.cdr is None:
            return None
        cdr = np.concatenate([self.cdr[c] for c in chains], axis=0)
        if encode:
            cdr = np.array([CDR_REVERSE[aa] for aa in cdr])
        return cdr

    def get_atom_mask(self, chains=None, cdr=None):
        """Get the atom mask of the protein.

        Parameters
        ----------
        chains : str, optional
            If specified, only the atom masks of the specified chains are returned (in the same order);
            otherwise, all atom masks are concatenated in alphabetical order of the chain IDs
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
        chains = self._get_chains_list(chains)
        seq = "".join([self.seq[c] for c in chains])
        atom_mask = np.concatenate([ATOM_MASKS[aa] for aa in seq])
        atom_mask[self.mask == 0] = 0
        if cdr is not None:
            atom_mask = atom_mask[self.cdr == cdr]
        return atom_mask

    @staticmethod
    def decode_cdr(cdr):
        """Decode the CDR information.

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
        """Decode the amino acid sequence.

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
    def from_dict(dictionary):
        """Load a protein entry from a dictionary.

        Parameters
        ----------
        dictionary : dict
            A nested dictionary where first-level keys are chain IDs and
            second-level keys are the following:
            - `'seq'` : amino acid sequence (one-letter code)
            - `'crd_bb'` : backbone coordinates, shaped `(L, 4, 3)`
            - `'crd_sc'` : sidechain coordinates, shaped `(L, 10, 3)`
            - `'msk'` : mask array where 1 indicates residues with known coordinates and 0
                indicates missing values, shaped `(L,)`
            - `'cdr'` (optional): CDR information, shaped `(L,)` encoded as integers where each
                integer corresponds to the index of the CDR type in
                `proteinflow.constants.CDR_ALPHABET`

        Returns
        -------
        entry : ProteinEntry
            A `ProteinEntry` object

        """
        chains = sorted(dictionary.keys())
        seq = [dictionary[k]["seq"] for k in chains]
        crd = [
            np.concatenate([dictionary[k]["crd_bb"], dictionary[k]["crd_sc"]], axis=1)
            for k in chains
        ]
        mask = [dictionary[k]["msk"] for k in chains]
        cdr = [dictionary[k].get("cdr", None) for k in chains]
        return ProteinEntry(seqs=seq, crds=crd, masks=mask, cdrs=cdr, chain_ids=chains)

    @staticmethod
    def from_pdb_entry(pdb_entry):
        """Load a protein entry from a `PDBEntry` object.

        Parameters
        ----------
        pdb_entry : PDBEntry
            A `PDBEntry` object

        Returns
        -------
        entry : ProteinEntry
            A `ProteinEntry` object

        """
        pdb_dict = {}
        fasta_dict = pdb_entry.get_fasta()
        for (chain,) in pdb_entry.get_chains():
            pdb_dict[chain] = {}
            fasta_seq = fasta_dict[chain]

            # align fasta and pdb and check criteria)
            mask = pdb_entry.get_mask([chain])[chain]
            if isinstance(pdb_entry, SAbDabEntry):
                pdb_dict[chain]["cdr"] = pdb_entry.get_cdr([chain])[chain]
            pdb_dict[chain]["seq"] = fasta_seq
            pdb_dict[chain]["msk"] = mask

            # go over rows of coordinates
            crd_arr = pdb_entry.get_coordinates_array(chain)

            pdb_dict[chain]["crd_bb"] = crd_arr[:, :4, :]
            pdb_dict[chain]["crd_sc"] = crd_arr[:, 4:, :]
            pdb_dict[chain]["msk"][
                (pdb_dict[chain]["crd_bb"] == 0).sum(-1).sum(-1) > 0
            ] = 0
        return ProteinEntry.from_dict(pdb_dict)

    @staticmethod
    def from_pdb(
        pdb_path,
        fasta_path=None,
        heavy_chain=None,
        light_chain=None,
        antigen_chains=None,
    ):
        """Load a protein entry from a PDB file.

        Parameters
        ----------
        pdb_path : str
            Path to the PDB file
        fasta_path : str, optional
            Path to the FASTA file; if not specified, the sequence is extracted
            from the PDB file
        heavy_chain : str, optional
            Chain ID of the heavy chain (to load a SAbDab entry)
        light_chain : str, optional
            Chain ID of the light chain (to load a SAbDab entry)
        antigen_chains : list of str, optional
            Chain IDs of the antigen chains (to load a SAbDab entry)

        Returns
        -------
        entry : ProteinEntry
            A `ProteinEntry` object

        """
        if heavy_chain is not None or light_chain is not None:
            pdb_entry = SAbDabEntry(
                pdb_path=pdb_path,
                fasta_path=fasta_path,
                heavy_chain=heavy_chain,
                light_chain=light_chain,
                antigen_chains=antigen_chains,
            )
        else:
            pdb_entry = PDBEntry(pdb_path=pdb_path, fasta_path=fasta_path)
        return ProteinEntry.from_pdb_entry(pdb_entry)

    @staticmethod
    def from_id(
        pdb_id,
        local_folder=".",
        heavy_chain=None,
        light_chain=None,
        antigen_chains=None,
    ):
        """Load a protein entry from a PDB file.

        Parameters
        ----------
        pdb_id : str
            PDB ID of the protein
        local_folder : str, default "."
            Path to the local folder where the PDB file is saved
        heavy_chain : str, optional
            Chain ID of the heavy chain (to load a SAbDab entry)
        light_chain : str, optional
            Chain ID of the light chain (to load a SAbDab entry)
        antigen_chains : list of str, optional
            Chain IDs of the antigen chains (to load a SAbDab entry)

        Returns
        -------
        entry : ProteinEntry
            A `ProteinEntry` object

        """
        if heavy_chain is not None or light_chain is not None:
            pdb_entry = SAbDabEntry.from_id(
                pdb_id=pdb_id,
                local_folder=local_folder,
                heavy_chain=heavy_chain,
                light_chain=light_chain,
                antigen_chains=antigen_chains,
            )
        else:
            pdb_entry = PDBEntry.from_id(pdb_id=pdb_id)
        return ProteinEntry.from_pdb_entry(pdb_entry)

    @staticmethod
    def from_pickle(path):
        """Load a protein entry from a pickle file.

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
        return ProteinEntry.from_dict(data)

    def to_dict(self):
        """Convert a protein entry into a dictionary.

        Returns
        -------
        dictionary : dict
            A nested dictionary where first-level keys are chain IDs and
            second-level keys are the following:
            - `'seq'` : amino acid sequence (one-letter code)
            - `'crd_bb'` : backbone coordinates, shaped `(L, 4, 3)`
            - `'crd_sc'` : sidechain coordinates, shaped `(L, 10, 3)`
            - `'msk'` : mask array where 1 indicates residues with known coordinates and 0
                indicates missing values, shaped `(L,)`
            - `'cdr'` (optional): CDR information, shaped `(L,)` encoded as integers where each
                integer corresponds to the index of the CDR type in
                `proteinflow.constants.CDR_ALPHABET`

        """
        data = {}
        for chain in self.get_chains():
            data[chain] = {
                "seq": self.seq[chain],
                "crd_bb": self.crd[chain][:, :4],
                "crd_sc": self.crd[chain][:, 4:],
                "msk": self.mask[chain],
            }
            if self.cdr[chain] is not None:
                data[chain]["cdr"] = self.cdr[chain]
        return data

    def to_pdb(
        self,
        path,
        only_ca=False,
        skip_oxygens=False,
        only_backbone=False,
        title="Untitled",
    ):
        """Save the protein entry to a PDB file.

        Parameters
        ----------
        path : str
            Path to the output PDB file
        only_ca : bool, default `False`
            If `True`, only backbone atoms are saved
        skip_oxygens : bool, default `False`
            If `True`, oxygen atoms are not saved
        only_backbone : bool, default `False`
            If `True`, only backbone atoms are saved
        title : str, default 'Untitled'
            Title of the PDB file

        """
        pdb_builder = PDBBuilder(
            self,
            only_ca=only_ca,
            skip_oxygens=skip_oxygens,
            only_backbone=only_backbone,
        )
        pdb_builder.save_pdb(path, title=title)

    def to_pickle(self, path):
        """Save a protein entry to a pickle file.

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
        data = self.to_dict()
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def dihedral_angles(self, chains=None):
        """Calculate the backbone dihedral angles (phi, psi) of the protein.

        Parameters
        ----------
        chains : list of str, optional
            If specified, only the dihedral angles of the specified chains are returned (in the same order);
            otherwise, all features are concatenated in alphabetical order of the chain IDs

        Returns
        -------
        angles : np.ndarray
            A `'numpy'` array of shape `(L, 2)` with backbone dihedral angles
            (phi, psi) in degrees; missing values are marked with zeros
        chains : list of str, optional
            If specified, only the dihedral angles of the specified chains are returned (in the same order);
            otherwise, all features are concatenated in alphabetical order of the chain IDs

        """
        angles = []
        chains = self._get_chains_list(chains)
        # N, C, Ca, O
        # psi
        for chain in chains:
            chain_angles = []
            crd = self.get_coordinates([chain])
            mask = self.get_mask([chain])
            p = crd[:-1, [0, 2, 1], :]
            p = np.concatenate([p, crd[1:, [0], :]], 1)
            p = np.pad(p, ((0, 1), (0, 0), (0, 0)))
            chain_angles.append(_dihedral_angle(p, mask))
            # phi
            p = crd[:-1, [1], :]
            p = np.concatenate([p, crd[1:, [0, 2, 1]]], 1)
            p = np.pad(p, ((1, 0), (0, 0), (0, 0)))
            chain_angles.append(_dihedral_angle(p, mask))
            angles.append(np.stack(chain_angles, -1))
        angles = np.concatenate(angles, 0)
        return angles

    def secondary_structure(self, chains=None):
        """Calculate the secondary structure of the protein.

        Parameters
        ----------
        chains : list of str, optional
            If specified, only the secondary structure of the specified chains is returned (in the same order);
            otherwise, all features are concatenated in alphabetical order of the chain IDs

        Returns
        -------
        sse : np.ndarray
            A `'numpy'` array of shape `(L, 3)` with secondary structure
            elements encoded as one-hot vectors (alpha-helix, beta-sheet, loop);
            missing values are marked with zeros
        chains : list of str, optional
            If specified, only the secondary structure of the specified chains is returned (in the same order);
            otherwise, all features are concatenated in alphabetical order of the chain IDs

        """
        chains = self._get_chains_list(chains)
        out = []
        for chain in chains:
            crd = self.get_coordinates([chain])
            sse_map = {"c": [0, 0, 1], "b": [0, 1, 0], "a": [1, 0, 0], "": [0, 0, 0]}
            sse = _annotate_sse(crd[:, :4])
            out += [sse_map[x] for x in sse]
        sse = np.array(out)
        return sse

    def sidechain_coordinates(self, chains=None):
        """Get the sidechain coordinates of the protein.

        Parameters
        ----------
        chains : list of str, optional
            If specified, only the sidechain coordinates of the specified chains is returned (in the same order);
            otherwise, all features are concatenated in alphabetical order of the chain IDs

        Returns
        -------
        crd : np.ndarray
            A `'numpy'` array of shape `(L, 10, 3)` with sidechain atom
            coordinates (check `proteinflow.sidechain_order()` for the order of
            atoms); missing values are marked with zeros
        chains : list of str, optional
            If specified, only the sidechain coordinates of the specified chains are returned (in the same order);
            otherwise, all features are concatenated in alphabetical order of the chain IDs

        """
        chains = self._get_chains_list(chains)
        return self.get_coordinates(chains)[:, 4:, :]

    def chemical_features(self, chains=None):
        """Calculate chemical features of the protein.

        Parameters
        ----------
        chains : list of str, optional
            If specified, only the chemical features of the specified chains is returned (in the same order);
            otherwise, all features are concatenated in alphabetical order of the chain IDs

        Returns
        -------
        features : np.ndarray
            A `'numpy'` array of shape `(L, 4)` with chemical features of the
            protein (hydropathy, volume, charge, polarity, acceptor/donor); missing
            values are marked with zeros
        chains : list of str, optional
            If specified, only the chemical features of the specified chains are returned (in the same order);
            otherwise, all features are concatenated in alphabetical order of the chain IDs

        """
        chains = self._get_chains_list(chains)
        seq = "".join([self.seq[chain] for chain in chains])
        features = np.array([_PMAP(x) for x in seq])
        return features

    def sidechain_orientation(self, chains=None):
        """Calculate the (global) sidechain orientation of the protein.

        Parameters
        ----------
        chains : list of str, optional
            If specified, only the sidechain orientation of the specified chains is returned (in the same order);
            otherwise, all features are concatenated in alphabetical order of the chain IDs

        Returns
        -------
        orientation : np.ndarray
            A `'numpy'` array of shape `(L, 3)` with sidechain orientation
            vectors; missing values are marked with zeros
        chains : list of str, optional
            If specified, only the sidechain orientation of the specified chains is returned (in the same order);
            otherwise, all features are concatenated in alphabetical order of the chain IDs

        """
        chains = self._get_chains_list(chains)
        crd = self.get_coordinates(chains=chains)
        crd_bb, crd_sc = crd[:, :4, :], crd[:, 4:, :]
        seq = self.get_sequence(chains=chains, encode=True)
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

    @lru_cache()
    def is_valid_pair(self, chain1, chain2, cutoff=10):
        """Check if two chains are a valid pair based on the distance between them.

        We consider two chains to be a valid pair if the distance between them is
        smaller than `cutoff` Angstroms. The distance is calculated as the minimum
        distance between any two atoms of the two chains.

        Parameters
        ----------
        chain1 : str
            Chain ID of the first chain
        chain2 : str
            Chain ID of the second chain
        cutoff : int, optional
            Minimum distance between the two chains (in Angstroms)

        Returns
        -------
        valid : bool
            `True` if the two chains are a valid pair, `False` otherwise

        """
        margin = cutoff * 3
        assert chain1 in self.get_chains(), f"Chain {chain1} not found"
        assert chain2 in self.get_chains(), f"Chain {chain2} not found"
        X1 = self.get_coordinates(chains=[chain1], only_known=True)
        X2 = self.get_coordinates(chains=[chain2], only_known=True)
        intersect_dim_X1 = []
        intersect_dim_X2 = []
        intersect_X1 = np.zeros(len(X1))
        intersect_X2 = np.zeros(len(X2))
        for dim in range(3):
            min_dim_1 = X1[:, 2, dim].min()
            max_dim_1 = X1[:, 2, dim].max()
            min_dim_2 = X2[:, 2, dim].min()
            max_dim_2 = X2[:, 2, dim].max()
            intersect_dim_X1.append(
                np.where(
                    np.logical_and(
                        X1[:, 2, dim] >= min_dim_2 - margin,
                        X1[:, 2, dim] <= max_dim_2 + margin,
                    )
                )[0]
            )
            intersect_dim_X2.append(
                np.where(
                    np.logical_and(
                        X2[:, 2, dim] >= min_dim_1 - margin,
                        X2[:, 2, dim] <= max_dim_1 + margin,
                    )
                )[0]
            )

        intersect_X1 = np.intersect1d(
            np.intersect1d(intersect_dim_X1[0], intersect_dim_X1[1]),
            intersect_dim_X1[2],
        )
        intersect_X2 = np.intersect1d(
            np.intersect1d(intersect_dim_X2[0], intersect_dim_X2[1]),
            intersect_dim_X2[2],
        )

        not_end_mask1 = np.where((X1[:, 2, :] == 0).sum(-1) != 3)[0]
        not_end_mask2 = np.where((X2[:, 2, :] == 0).sum(-1) != 3)[0]

        intersect_X1 = np.intersect1d(intersect_X1, not_end_mask1)
        intersect_X2 = np.intersect1d(intersect_X2, not_end_mask2)

        diff = X1[intersect_X1, 2, np.newaxis, :] - X2[intersect_X2, 2, :]
        distances = np.sqrt(np.sum(diff**2, axis=2))

        if np.sum(distances < cutoff) < 3:
            return False
        else:
            return True

    def get_index_array(self, chains=None, index_bump=100):
        """Get the index array of the protein.

        The index array is a `'numpy'` array of shape `(L,)` with the index of each residue along the chain.

        Parameters
        ----------
        chains : list of str, optional
            If specified, only the index array of the specified chains is returned (in the same order);
            otherwise, all features are concatenated in alphabetical order of the chain IDs
        index_bump : int, default 0
            If specified, the index is bumped by this number between chains

        Returns
        -------
        index_array : np.ndarray
            A `'numpy'` array of shape `(L,)` with the index of each residue along the chain; if multiple chains
            are specified, the index is bumped by `index_bump` at the beginning of each chain

        """
        chains = self._get_chains_list(chains)
        start_value = 0
        start_index = 0
        index_array = np.zeros(self.get_length(chains))
        for chain in chains:
            chain_length = self.get_length([chain])
            index_array[start_index : start_index + chain_length] = np.arange(
                start_value, start_value + chain_length
            )
            start_value += chain_length + index_bump
            start_index += chain_length
        return index_array.astype(int)

    def get_chain_id_dict(self, chains=None):
        """Get the dictionary mapping from chain indices to chain IDs.

        Parameters
        ----------
        chains : list of str, optional
            If specified, only the chain IDs of the specified chains are returned

        Returns
        -------
        chain_id_dict : dict
            A dictionary mapping from chain indices to chain IDs

        """
        chains = self._get_chains_list(chains)
        chain_id_dict = {x: i for i, x in enumerate(self.get_chains()) if x in chains}
        return chain_id_dict

    def get_chain_id_array(self, chains=None, encode=True):
        """Get the chain ID array of the protein.

        The chain ID array is a `'numpy'` array of shape `(L,)` with the chain ID of each residue.
        The chain ID is the index of the chain in the alphabetical order of the chain IDs. To get a
        mapping from the index to the chain ID, use `get_chain_id_dict()`.

        Parameters
        ----------
        chains : list of str, optional
            If specified, only the chain ID array of the specified chains is returned (in the same order);
            otherwise, all features are concatenated in alphabetical order of the chain IDs
        encode : bool, default True
            If True, the chain ID is encoded as an integer; otherwise, the chain ID is the chain ID string

        Returns
        -------
        chain_id_array : np.ndarray
            A `'numpy'` array of shape `(L,)` with the chain ID of each residue

        """
        id_dict = self.get_chain_id_dict()
        if encode:
            index_array = np.zeros(self.get_length(chains))
        else:
            index_array = np.empty(self.get_length(chains), dtype=object)
        start_index = 0
        for chain in self._get_chains_list(chains):
            chain_length = self.get_length([chain])
            index_array[start_index : start_index + chain_length] = (
                id_dict[chain] if encode else chain
            )
            start_index += chain_length
        return index_array

    def _get_highlight_mask_dict(self, highlight_mask=None):
        chain_arr = self.get_chain_id_array(encode=False)
        mask_arr = self.get_mask().astype(bool)
        highlight_mask_dict = {}
        if highlight_mask is not None:
            chains = self.get_chains()
            for chain in chains:
                chain_mask = chain_arr == chain
                pdb_highlight = highlight_mask[mask_arr & chain_mask]
                highlight_mask_dict[chain] = pdb_highlight
        return highlight_mask_dict

    def _get_atom_dicts(self, highlight_mask=None, style="cartoon"):
        """Get the atom dictionaries of the protein."""
        highlight_mask_dict = self._get_highlight_mask_dict(highlight_mask)
        with tempfile.NamedTemporaryFile(suffix=".pdb") as tmp:
            self.to_pdb(tmp.name)
            pdb_entry = PDBEntry(tmp.name)
        return pdb_entry._get_atom_dicts(
            highlight_mask_dict=highlight_mask_dict, style=style
        )

    def visualize(self, highlight_mask=None, style="cartoon"):
        """Visualize the protein in a notebook.

        Parameters
        ----------
        highlight_mask : np.ndarray, optional
            A `'numpy'` array of shape `(L,)` with the residues to highlight
            marked with 1 and the rest marked with 0
        style : str, default 'cartoon'
            The style of the visualization; one of 'cartoon', 'sphere', 'stick', 'line', 'cross'

        """
        highlight_mask_dict = self._get_highlight_mask_dict(highlight_mask)
        with tempfile.NamedTemporaryFile(suffix=".pdb") as tmp:
            self.to_pdb(tmp.name)
            pdb_entry = PDBEntry(tmp.name)
        pdb_entry.visualize(highlight_mask_dict=highlight_mask_dict)


class PDBEntry:
    """A class for parsing PDB entries."""

    def __init__(self, pdb_path, fasta_path=None):
        """Initialize a PDBEntry object.

        If no FASTA path is provided, the sequences will be fully inferred
        from the PDB file.

        Parameters
        ----------
        pdb_path : str
            Path to the PDB file
        fasta_path : str, optional
            Path to the FASTA file

        """
        self.pdb_path = pdb_path
        self.fasta_path = fasta_path
        self.pdb_id = os.path.basename(pdb_path).split(".")[0].split("-")[0]
        self.crd_df, self.seq_df = self._parse_structure()
        self.fasta_dict = self._parse_fasta()

    @staticmethod
    def from_id(pdb_id, local_folder="."):
        """Initialize a `PDBEntry` object from a PDB Id.

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

    def _get_relevant_chains(self):
        """Get the chains that are included in the entry."""
        return list(self.seq_df["chain_id"].unique())

    @staticmethod
    def parse_fasta(fasta_path):
        """Read a fasta file.

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

    def _parse_fasta(self):
        """Parse the fasta file."""
        # download fasta and check if it contains only proteins
        chains = self._get_relevant_chains()
        if self.fasta_path is None:
            seqs_dict = {k: self._pdb_sequence(k, suppress_check=True) for k in chains}
        else:
            seqs_dict = self.parse_fasta(self.fasta_path)
        # retrieve sequences that are relevant for this PDB from the fasta file
        seqs_dict = {k.upper(): v for k, v in seqs_dict.items()}
        if all([len(x) == 3 and len(set(list(x))) == 1 for x in seqs_dict.keys()]):
            seqs_dict = {k[0]: v for k, v in seqs_dict.items()}

        if not {x.split("-")[0].upper() for x in chains}.issubset(
            set(list(seqs_dict.keys()))
        ):
            raise PDBError("Some chains in the PDB do not appear in the fasta file")

        fasta_dict = {k: seqs_dict[k.split("-")[0].upper()] for k in chains}
        return fasta_dict

    def _parse_structure(self, chains=None):
        """Parse the structure of the protein."""
        cif = self.pdb_path.endswith("cif.gz")

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
        crd_df = crd_df[crd_df["record_name"] == "ATOM"].reset_index()
        if "insertion" in crd_df.columns:
            crd_df["unique_residue_number"] = crd_df.apply(
                lambda row: f"{row['residue_number']}_{row['insertion']}", axis=1
            )
        seq_df = p.amino3to1()

        return crd_df, seq_df

    def _get_chain(self, chain):
        """Check the chain ID."""
        if chain is None:
            return chain
        if chain not in self.get_chains():
            raise PDBError("Chain not found")
        return chain

    def get_pdb_df(self, chain=None):
        """Return the PDB dataframe.

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
        chain = self._get_chain(chain)
        if chain is None:
            return self.crd_df
        else:
            return self.crd_df[self.crd_df["chain_id"] == chain]

    def get_sequence_df(self, chain=None, suppress_check=False):
        """Return the sequence dataframe.

        If `chain` is provided, only information for this chain is returned.

        Parameters
        ----------
        chain : str, optional
            Chain identifier
        suppress_check : bool, default False
            If True, do not check if the chain is in the PDB

        Returns
        -------
        df : pd.DataFrame
            A dataframe containing the sequence and chain information
            (analogous to the `BioPandas.pdb.PandasPdb.amino3to1` method output)

        """
        if not suppress_check:
            chain = self._get_chain(chain)
        if chain is None:
            return self.seq_df
        else:
            return self.seq_df[self.seq_df["chain_id"] == chain]

    def get_fasta(self):
        """Return the fasta dictionary.

        Returns
        -------
        fasta_dict : dict
            A dictionary containing all the (author) chains in a fasta file (keys)
            and their corresponding sequence (values)

        """
        return self.fasta_dict

    def get_chains(self):
        """Return the chains in the PDB.

        Returns
        -------
        chains : list
            A list of chain identifiers

        """
        return list(self.fasta_dict.keys())

    @lru_cache()
    def _pdb_sequence(self, chain, suppress_check=False):
        """Return the PDB sequence for a given chain ID."""
        return "".join(
            self.get_sequence_df(chain, suppress_check=suppress_check)["residue_name"]
        )

    @lru_cache()
    def _align_chain(self, chain):
        """Align the PDB sequence to the FASTA sequence for a given chain ID."""
        chain = self._get_chain(chain)
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

    def get_alignment(self, chains=None):
        """Return the alignment between the PDB and the FASTA sequence.

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
        return {chain: self._align_chain(chain)[0] for chain in chains}

    def get_mask(self, chains=None):
        """Return the mask of the alignment between the PDB and the FASTA sequence.

        Parameters
        ----------
        chains : list, optional
            A list of chain identifiers (if not provided, all chains are aligned)

        Returns
        -------
        mask : dict
            A dictionary containing the `np.ndarray` mask for each chain (0 where the
            aligned sequence has gaps and 1 where it does not)

        """
        alignment = self.get_alignment(chains)
        return {
            chain: (np.array(list(seq)) != "-").astype(int)
            for chain, seq in alignment.items()
        }

    def has_unnatural_amino_acids(self, chains=None):
        """Check if the PDB contains unnatural amino acids.

        Parameters
        ----------
        chains : list, optional
            A list of chain identifiers (if not provided, all chains are checked)

        Returns
        -------
        bool
            True if the PDB contains unnatural amino acids, False otherwise

        """
        if chains is None:
            chains = [None]
        for chain in chains:
            crd = self.get_pdb_df(chain)
            if not crd["residue_name"].isin(D3TO1.keys()).all():
                return True
        return False

    def get_coordinates_array(self, chain):
        """Return the coordinates of the PDB as a numpy array.

        The atom order is the same as in the `ProteinEntry.ATOM_ORDER` dictionary.
        The array has zeros where the mask has zeros and that is where the sequence
        alignment to the FASTA has gaps (unknown coordinates).

        Parameters
        ----------
        chain : str
            Chain identifier

        Returns
        -------
        crd_arr : np.ndarray
            A numpy array of shape (n_residues, 14, 3) containing the coordinates
            of the PDB (zeros where the coordinates are unknown)

        """
        chain_crd = self.get_pdb_df(chain)

        # align fasta and pdb and check criteria)
        mask = self.get_mask([chain])[chain]

        # go over rows of coordinates
        crd_arr = np.zeros((len(mask), 14, 3))

        def arr_index(row):
            atom = row["atom_name"]
            if atom.startswith("H") or atom == "OXT":
                return -1  # ignore hydrogens and OXT
            order = ProteinEntry.ATOM_ORDER[row["residue_name"]]
            try:
                return order.index(atom)
            except ValueError:
                raise PDBError(f"Unexpected atoms ({atom})")

        indices = chain_crd.apply(arr_index, axis=1)
        indices = indices.astype(int)
        informative_mask = indices != -1
        res_indices = np.where(mask == 1)[0]
        unique_numbers = self.get_unique_residue_numbers(chain)
        pdb_seq = self._pdb_sequence(chain)
        if len(unique_numbers) != len(pdb_seq):
            raise PDBError("Inconsistencies in the biopandas dataframe")
        replace_dict = {x: y for x, y in zip(unique_numbers, res_indices)}
        chain_crd.loc[:, "unique_residue_number"] = chain_crd[
            "unique_residue_number"
        ].replace(replace_dict)
        crd_arr[
            chain_crd[informative_mask]["unique_residue_number"].astype(int),
            indices[informative_mask],
        ] = chain_crd[informative_mask][["x_coord", "y_coord", "z_coord"]]
        return crd_arr

    def get_unique_residue_numbers(self, chain):
        """Return the unique residue numbers (residue number + insertion code).

        Parameters
        ----------
        chain : str
            Chain identifier

        Returns
        -------
        unique_numbers : list
            A list of unique residue numbers

        """
        return self.get_pdb_df(chain)["unique_residue_number"].unique().tolist()

    def _get_atom_dicts(self, highlight_mask_dict=None, style="cartoon"):
        """Get the atom dictionaries for visualization."""
        assert style in ["cartoon", "sphere", "stick", "line", "cross"]
        outstr = []
        df_ = self.crd_df.sort_values(["chain_id", "residue_number"], inplace=False)
        for _, row in df_.iterrows():
            outstr.append(_Atom(row))
        chains = self.get_chains()
        colors = {ch: COLORS[i % len(COLORS)] for i, ch in enumerate(chains)}
        chain_counters = defaultdict(int)
        chain_last_res = defaultdict(lambda: None)
        if highlight_mask_dict is not None:
            for chain, mask in highlight_mask_dict.items():
                assert len(mask) == len(
                    self._pdb_sequence(chain)
                ), "Mask length does not match sequence length"
        for at in outstr:
            if at["resid"] != chain_last_res[at["chain"]]:
                chain_last_res[at["chain"]] = at["resid"]
                chain_counters[at["chain"]] += 1
            at["pymol"] = {style: {"color": colors[at["chain"]]}}
            if highlight_mask_dict is not None and at["chain"] in highlight_mask_dict:
                num = chain_counters[at["chain"]]
                if highlight_mask_dict[at["chain"]][num - 1] == 1:
                    at["pymol"] = {style: {"color": "red"}}
        return outstr

    def visualize(self, highlight_mask_dict=None, style="cartoon"):
        """Visualize the protein in a notebook.

        Parameters
        ----------
        highlight_mask_dict : dict, optional
            A dictionary mapping from chain IDs to a mask of 0s and 1s of the same length as the chain sequence;
            the atoms corresponding to 1s will be highlighted in red
        style : str, default 'cartoon'
            The style of the visualization; one of 'cartoon', 'sphere', 'stick', 'line', 'cross'

        """
        outstr = self._get_atom_dicts(highlight_mask_dict, style=style)
        vis_string = "".join([str(x) for x in outstr])
        view = py3Dmol.view(width=400, height=300)
        view.addModelsAsFrames(vis_string)
        for i, at in enumerate(outstr):
            view.setStyle(
                {"model": -1, "serial": i + 1},
                at["pymol"],
            )
        view.zoomTo()
        view.show()


class SAbDabEntry(PDBEntry):
    """A class for parsing SAbDab entries."""

    def __init__(
        self,
        pdb_path,
        fasta_path,
        heavy_chain=None,
        light_chain=None,
        antigen_chains=None,
    ):
        """Initialize the SAbDabEntry.

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
        self.reverse_chain_dict = {
            heavy_chain: "heavy",
            light_chain: "light",
        }
        for antigen_chain in antigen_chains:
            self.reverse_chain_dict[antigen_chain] = "antigen"
        super().__init__(pdb_path, fasta_path)

    def _get_relevant_chains(self):
        """Get the chains that are included in the entry."""
        chains = []
        if self.chain_dict["heavy"] is not None:
            chains.append(self.chain_dict["heavy"])
        if self.chain_dict["light"] is not None:
            chains.append(self.chain_dict["light"])
        chains.extend(self.chain_dict["antigen"])
        return chains

    @staticmethod
    def from_id(
        pdb_id,
        local_folder=".",
        light_chain=None,
        heavy_chain=None,
        antigen_chains=None,
    ):
        """Create a SAbDabEntry from a PDB ID.

        Either the light or the heavy chain must be provided.

        Parameters
        ----------
        pdb_id : str
            PDB ID
        local_folder : str, optional
            Local folder to download the PDB and FASTA files
        light_chain : str, optional
            Light chain identifier (author chain name)
        heavy_chain : str, optional
            Heavy chain identifier (author chain name)
        antigen_chains : list, optional
            List of antigen chain identifiers (author chain names)

        Returns
        -------
        entry : SAbDabEntry
            A SAbDabEntry object

        """
        pdb_path = download_pdb(pdb_id, local_folder, sabdab=True)
        fasta_path = download_fasta(pdb_id, local_folder)
        return SAbDabEntry(
            pdb_path=pdb_path,
            fasta_path=fasta_path,
            light_chain=light_chain,
            heavy_chain=heavy_chain,
            antigen_chains=antigen_chains,
        )

    def _get_chain(self, chain):
        """Return the chain identifier."""
        if chain in ["heavy", "light"]:
            chain = self.chain_dict[chain]
        return super()._get_chain(chain)

    def heavy_chain(self):
        """Return the heavy chain identifier.

        Returns
        -------
        chain : str
            The heavy chain identifier

        """
        return self.chain_dict["heavy"]

    def light_chain(self):
        """Return the light chain identifier.

        Returns
        -------
        chain : str
            The light chain identifier

        """
        return self.chain_dict["light"]

    def antigen_chains(self):
        """Return the antigen chain identifiers.

        Returns
        -------
        chains : list
            The antigen chain identifiers

        """
        return self.chain_dict["antigen"]

    def chains(self):
        """Return the chains in the PDB.

        Returns
        -------
        chains : list
            A list of chain identifiers

        """
        return [self.heavy_chain(), self.light_chain()] + self.antigen_chains()

    def chain_type(self, chain):
        """Return the type of a chain.

        Parameters
        ----------
        chain : str
            Chain identifier

        Returns
        -------
        chain_type : str
            The type of the chain (heavy, light or antigen)

        """
        if chain in self.reverse_chain_dict:
            return self.reverse_chain_dict[chain]
        raise PDBError("Chain not found")

    @lru_cache()
    def _get_chain_cdr(self, chain, align_to_fasta=True):
        """Return the CDRs for a given chain ID."""
        chain = self._get_chain(chain)
        chain_crd = self.get_pdb_df(chain)
        chain_type = self.chain_type(chain)[0].upper()
        pdb_seq = self._pdb_sequence(chain)
        unique_numbers = chain_crd["unique_residue_number"].unique()
        if len(unique_numbers) != len(pdb_seq):
            raise PDBError("Inconsistencies in the biopandas dataframe")
        if chain_type in ["H", "L"]:
            cdr_arr = [
                CDR_VALUES[chain_type][int(x.split("_")[0])] for x in unique_numbers
            ]
            cdr_arr = np.array(cdr_arr)
        else:
            cdr_arr = np.array(["-"] * len(unique_numbers), dtype=object)
        if align_to_fasta:
            aligned_seq, _ = self._align_chain(chain)
            aligned_seq_arr = np.array(list(aligned_seq))
            cdr_arr_aligned = np.array(["-"] * len(aligned_seq), dtype=object)
            cdr_arr_aligned[aligned_seq_arr != "-"] = cdr_arr
            cdr_arr = cdr_arr_aligned
        return cdr_arr

    def get_cdr(self, chains=None):
        """Return CDR arrays.

        Parameters
        ----------
        chains : list, optional
            A list of chain identifiers (if not provided, all chains are processed)

        Returns
        -------
        cdrs : dict
            A dictionary containing the CDR arrays for each of the chains

        """
        if chains is None:
            chains = self.chains()
        return {chain: self._get_chain_cdr(chain) for chain in chains}
