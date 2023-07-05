import pickle

import numpy as np

from proteinflow.constants import (
    _PMAP,
    ALPHABET,
    ALPHABET_REVERSE,
    ATOM_MASKS,
    BACKBONE_ORDER,
    CDR_ALPHABET,
    CDR_REVERSE,
    MAIN_ATOM_DICT,
    SIDECHAIN_ORDER,
)
from proteinflow.data.utils import _annotate_sse, _dihedral_angle


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
