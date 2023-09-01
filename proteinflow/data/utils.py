"""Utility functions for working with protein data."""
import itertools
from copy import deepcopy

import numpy as np
import pandas as pd
from biopandas.mmcif import PandasMmcif
from biotite.structure.geometry import angle, dihedral, distance
from einops import rearrange

from proteinflow.constants import (
    ATOM_MAP_1,
    ATOM_MAP_3,
    ATOM_MAP_4,
    ATOM_MAP_14,
    D3TO1,
    GLOBAL_PAD_CHAR,
    ONE_TO_THREE_LETTER_MAP,
)


class PDBBuilder:
    """
    Creates a PDB file from a `ProteinEntry` object.

    Adapted from [sidechainnet](https://github.com/jonathanking/sidechainnet) by Jonathan King.
    """

    def __init__(
        self, protein_entry, only_ca=False, skip_oxygens=False, only_backbone=False
    ):
        """Initialize a PDBBuilder object.

        Parameters
        ----------
        protein_entry : ProteinEntry
            The protein entry to build a PDB file from
        only_ca : bool, default False
            If True, only the alpha carbon atoms will be included in the PDB file
        skip_oxygens : bool, default False
            If True, the oxygen atoms will be excluded from the PDB file
        only_backbone : bool, default False
            If True, only the backbone atoms will be included in the PDB file

        """
        seq = protein_entry.get_sequence()
        coords = protein_entry.get_coordinates()
        mask = protein_entry.get_mask().astype(bool)
        seq = "".join(np.array(list(seq))[mask])
        coords = coords[mask]
        if (coords[:, 4:] == 0).all():
            only_backbone = True
        if only_ca:
            coords = coords[:, 2, :].unsqueeze(1)
        elif skip_oxygens:
            coords = coords[:, :3, :]
        elif only_backbone:
            coords = coords[:, :4, :]
        coords = rearrange(coords, "l n c -> (l n) c")

        if only_ca:
            atoms_per_res = 1
        elif skip_oxygens:
            atoms_per_res = 3
        elif only_backbone:
            atoms_per_res = 4
        else:
            atoms_per_res = 14

        self.only_ca = only_ca
        self.skip_oxygens = skip_oxygens
        self.atoms_per_res = atoms_per_res
        self.only_backbone = only_backbone
        self.coords = coords
        self.seq = seq
        self.mapping = self._make_mapping_from_seq()

        self.chain_ids = protein_entry.get_chain_id_array(encode=False)[mask]
        self.chain_ids_unique = protein_entry.get_chains()

        # PDB Formatting Information
        self.format_str = (
            "{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}"
            "{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}"
        )
        self.defaults = {
            "alt_loc": "",
            "chain_id": "A",
            "insertion_code": "",
            "occupancy": 1,
            "temp_factor": 0,
            "element_sym": "",
            "charge": "",
        }
        self.title = "Untitled"
        self.atom_nbr = 1
        self.res_nbr = 1
        self._pdb_str = ""
        self._pdb_body_lines = []
        self._pdb_lines = []

    def _coord_generator(self):
        """Return a generator to iteratively yield self.atoms_per_res atoms at a time."""
        atoms_per_res = self.atoms_per_res
        remove_padding = False
        coords = self.coords
        coord_idx = 0
        while coord_idx < coords.shape[0]:
            _slice = coords[coord_idx : coord_idx + atoms_per_res]
            if remove_padding:
                non_pad_locs = (_slice != GLOBAL_PAD_CHAR).any(axis=1)
                _slice = _slice[non_pad_locs]
            yield _slice
            coord_idx += atoms_per_res

    def _get_line_for_atom(
        self, res_name, atom_name, atom_coords, chain_id, missing=False
    ):
        """Return the 'ATOM...' line in PDB format for the specified atom.

        If missing, this function should have special, but not yet determined,
        behavior.

        """
        if missing:
            occupancy = 0
        else:
            occupancy = self.defaults["occupancy"]
        return self.format_str.format(
            "ATOM",
            self.atom_nbr,
            atom_name,
            self.defaults["alt_loc"],
            ONE_TO_THREE_LETTER_MAP[res_name],
            chain_id,
            self.res_nbr,
            self.defaults["insertion_code"],
            atom_coords[0],
            atom_coords[1],
            atom_coords[2],
            occupancy,
            self.defaults["temp_factor"],
            atom_name[0],
            self.defaults["charge"],
        )

    def _get_lines_for_residue(
        self, res_name, atom_names, coords, chain_id, n_terminal=False, c_terminal=False
    ):
        """Return a list of PDB-formatted lines for all atoms in a single residue.

        Calls get_line_for_atom.

        """
        residue_lines = []
        for atom_name, atom_coord in zip(atom_names, coords):
            if (
                atom_name == "PAD"
                or np.isnan(atom_coord).sum() > 0
                or atom_coord.sum() == 0
            ):
                continue
            residue_lines.append(
                self._get_line_for_atom(res_name, atom_name, atom_coord, chain_id)
            )
            self.atom_nbr += 1

        return residue_lines

    def _get_lines_for_protein(self):
        """Return a list of PDB-formatted lines for all residues in this protein.

        Calls get_lines_for_residue.

        """
        self._pdb_body_lines = []
        self.res_nbr = 1
        self.atom_nbr = 1
        mapping_coords = zip(self.mapping, self._coord_generator())
        for index, ((res_name, atom_names), res_coords) in enumerate(mapping_coords):
            self._pdb_body_lines.extend(
                self._get_lines_for_residue(
                    res_name,
                    atom_names,
                    res_coords,
                    self.chain_ids[index],
                    n_terminal=index == 0,
                    c_terminal=index == len(self.seq) - 1,
                )
            )
            self.res_nbr += 1
        return self._pdb_body_lines

    def _make_header(self, title):
        """Return a string representing the PDB header."""
        return f"REMARK  {title}" + "\n" + self._make_SEQRES()

    @staticmethod
    def _make_footer():
        """Return a string representing the PDB footer."""
        return "TER\nEND          \n"

    def _make_mapping_from_seq(self):
        """Make an atom name mapping.

        Given a protein sequence, this returns a mapping that assumes coords are
        generated in groups of atoms_per_res (the output is L x atoms_per_res x 3).

        """
        if self.only_ca:
            atom_names = ATOM_MAP_1
        elif self.skip_oxygens:
            atom_names = ATOM_MAP_3
        elif self.only_backbone:
            atom_names = ATOM_MAP_4
        else:
            atom_names = ATOM_MAP_14
        mapping = []
        for residue in self.seq:
            mapping.append((residue, atom_names[residue]))
        return mapping

    def get_pdb_string(self, title=None):
        """Return a string representing the PDB file for this protein."""
        if not title:
            title = self.title

        if self._pdb_str:
            return self._pdb_str
        self._get_lines_for_protein()
        self._pdb_lines = (
            [self._make_header(title)] + self._pdb_body_lines + [self._make_footer()]
        )
        self._pdb_str = "\n".join(self._pdb_lines)
        return self._pdb_str

    def _make_SEQRES(self):
        """Return a SEQRES entry as a multi-line string for this PDB file."""
        seq = np.array(list(self.seq))
        lines = []
        for chain in self.chain_ids_unique:
            lineno = 1
            seq_chain = seq[self.chain_ids == chain]
            three_letter_seq = [ONE_TO_THREE_LETTER_MAP[c] for c in seq_chain]
            residue_blocks = list(_split_every(13, three_letter_seq))
            nres = len(self.seq)
            for block in residue_blocks:
                res_block_str = " ".join(block)
                cur_line = (
                    f"SEQRES {lineno: >3} {chain} {nres: >4}  {res_block_str: <61}"
                )
                lines.append(cur_line)
                lineno += 1
        return "\n".join(lines)

    def save_pdb(self, path, title="UntitledProtein"):
        """Write out the generated PDB file as a string to the specified path."""
        with open(path, "w") as outfile:
            outfile.write(self.get_pdb_string(title))


def _split_every(n, iterable):
    """Split iterable into chunks. From https://stackoverflow.com/a/1915307/2780645."""
    i = iter(iterable)
    piece = list(itertools.islice(i, n))
    while piece:
        yield piece
        piece = list(itertools.islice(i, n))


class PDBError(ValueError):
    """Class for errors related to PDB processing."""

    pass


def _annotate_sse(X):
    """
    Annotation of secondary structure elements (SSEs) in a chain.

    Adapted from [biotite](https://github.com/biotite-dev/biotite).

    """
    ca_coord = X[:, 2, :]
    length = ca_coord.shape[0]

    _r_helix = (np.deg2rad(89 - 12), np.deg2rad(89 + 12))
    _a_helix = (np.deg2rad(50 - 20), np.deg2rad(50 + 20))
    _d3_helix = ((5.3 - 0.5), (5.3 + 0.5))
    _d4_helix = ((6.4 - 0.6), (6.4 + 0.6))

    _r_strand = (np.deg2rad(124 - 14), np.deg2rad(124 + 14))
    _a_strand = (np.deg2rad(-180), np.deg2rad(-125), np.deg2rad(145), np.deg2rad(180))
    _d2_strand = ((6.7 - 0.6), (6.7 + 0.6))
    _d3_strand = ((9.9 - 0.9), (9.9 + 0.9))
    _d4_strand = ((12.4 - 1.1), (12.4 + 1.1))

    d2i = np.full(length, np.nan)
    d3i = np.full(length, np.nan)
    d4i = np.full(length, np.nan)
    ri = np.full(length, np.nan)
    ai = np.full(length, np.nan)

    d2i[1 : length - 1] = distance(ca_coord[0 : length - 2], ca_coord[2:length])
    d3i[1 : length - 2] = distance(ca_coord[0 : length - 3], ca_coord[3:length])
    d4i[1 : length - 3] = distance(ca_coord[0 : length - 4], ca_coord[4:length])
    ri[1 : length - 1] = angle(
        ca_coord[0 : length - 2], ca_coord[1 : length - 1], ca_coord[2:length]
    )
    ai[1 : length - 2] = dihedral(
        ca_coord[0 : length - 3],
        ca_coord[1 : length - 2],
        ca_coord[2 : length - 1],
        ca_coord[3 : length - 0],
    )

    sse = np.full(len(ca_coord), "c", dtype="U1")
    # Annotate helices
    # Find CA that meet criteria for potential helices
    is_pot_helix = np.zeros(len(sse), dtype=bool)
    for i in range(len(sse)):
        if (
            d3i[i] >= _d3_helix[0]
            and d3i[i] <= _d3_helix[1]
            and d4i[i] >= _d4_helix[0]
            and d4i[i] <= _d4_helix[1]
        ) or (
            ri[i] >= _r_helix[0]
            and ri[i] <= _r_helix[1]
            and ai[i] >= _a_helix[0]
            and ai[i] <= _a_helix[1]
        ):
            is_pot_helix[i] = True
    # Real helices are 5 consecutive helix elements
    is_helix = np.zeros(len(sse), dtype=bool)
    counter = 0
    for i in range(len(sse)):
        if is_pot_helix[i]:
            counter += 1
        else:
            if counter >= 5:
                is_helix[i - counter : i] = True
            counter = 0
    # Extend the helices by one at each end if CA meets extension criteria
    i = 0
    while i < len(sse):
        if is_helix[i]:
            sse[i] = "a"
            if (d3i[i - 1] >= _d3_helix[0] and d3i[i - 1] <= _d3_helix[1]) or (
                ri[i - 1] >= _r_helix[0] and ri[i - 1] <= _r_helix[1]
            ):
                sse[i - 1] = "a"
            sse[i] = "a"
            if (d3i[i + 1] >= _d3_helix[0] and d3i[i + 1] <= _d3_helix[1]) or (
                ri[i + 1] >= _r_helix[0] and ri[i + 1] <= _r_helix[1]
            ):
                sse[i + 1] = "a"
        i += 1

    # Annotate sheets
    # Find CA that meet criteria for potential strands
    is_pot_strand = np.zeros(len(sse), dtype=bool)
    for i in range(len(sse)):
        if (
            d2i[i] >= _d2_strand[0]
            and d2i[i] <= _d2_strand[1]
            and d3i[i] >= _d3_strand[0]
            and d3i[i] <= _d3_strand[1]
            and d4i[i] >= _d4_strand[0]
            and d4i[i] <= _d4_strand[1]
        ) or (
            ri[i] >= _r_strand[0]
            and ri[i] <= _r_strand[1]
            and (
                (ai[i] >= _a_strand[0] and ai[i] <= _a_strand[1])
                or (ai[i] >= _a_strand[2] and ai[i] <= _a_strand[3])
            )
        ):
            is_pot_strand[i] = True
    # Real strands are 5 consecutive strand elements,
    # or shorter fragments of at least 3 consecutive strand residues,
    # if they are in hydrogen bond proximity to 5 other residues
    is_strand = np.zeros(len(sse), dtype=bool)
    counter = 0
    contacts = 0
    for i in range(len(sse)):
        if is_pot_strand[i]:
            counter += 1
            coord = ca_coord[i]
            for strand_coord in ca_coord:
                dist = distance(coord, strand_coord)
                if dist >= 4.2 and dist <= 5.2:
                    contacts += 1
        else:
            if counter >= 4:
                is_strand[i - counter : i] = True
            elif counter == 3 and contacts >= 5:
                is_strand[i - counter : i] = True
            counter = 0
            contacts = 0
    # Extend the strands by one at each end if CA meets extension criteria
    i = 0
    while i < len(sse):
        if is_strand[i]:
            sse[i] = "b"
            if d3i[i - 1] >= _d3_strand[0] and d3i[i - 1] <= _d3_strand[1]:
                sse[i - 1] = "b"
            sse[i] = "b"
            if d3i[i + 1] >= _d3_strand[0] and d3i[i + 1] <= _d3_strand[1]:
                sse[i + 1] = "b"
        i += 1
    return sse


def _dihedral_angle(crd, msk):
    """Compute the dihedral angle given coordinates."""
    p0 = crd[..., 0, :]
    p1 = crd[..., 1, :]
    p2 = crd[..., 2, :]
    p3 = crd[..., 3, :]

    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    b1 /= np.expand_dims(np.linalg.norm(b1, axis=-1), -1) + 1e-7

    v = b0 - np.expand_dims(np.einsum("bi,bi->b", b0, b1), -1) * b1
    w = b2 - np.expand_dims(np.einsum("bi,bi->b", b2, b1), -1) * b1

    x = np.einsum("bi,bi->b", v, w)
    y = np.einsum("bi,bi->b", np.cross(b1, v), w)
    dh = np.degrees(np.arctan2(y, x))
    dh[1 - msk] = 0
    return dh


class CustomMmcif(PandasMmcif):
    """
    A modification of `PandasMmcif`.

    Adds a `get_model` method and renames the columns to match PDB format.

    """

    def read_mmcif(self, path: str):
        """Read a PDB file in mmCIF format.

        Parameters
        ----------
        path : str
            Path to the file.

        Returns
        -------
        x : CustomMmcif
            The parsed file.

        """
        x = super().read_mmcif(path)
        x.df["ATOM"].rename(
            {
                "label_comp_id": "residue_name",
                "label_seq_id": "residue_number",
                "label_atom_id": "atom_name",
                "id": "atom_number",
                "group_PDB": "record_name",
                "Cartn_x": "x_coord",
                "Cartn_y": "y_coord",
                "Cartn_z": "z_coord",
                "pdbx_PDB_ins_code": "insertion",
                "type_symbol": "element_symbol",
            },
            axis=1,
            inplace=True,
        )
        x.df["ATOM"]["chain_id"] = x.df["ATOM"]["auth_asym_id"]
        return x

    def amino3to1(self):
        """Return a dataframe with the amino acid names converted to one letter codes."""
        tmp = self.df["ATOM"]
        cmp = "placeholder"
        indices = []

        residue_number_insertion = tmp["residue_number"].astype(str) + tmp["insertion"]

        for num, ind in zip(residue_number_insertion, np.arange(tmp.shape[0])):
            if num != cmp:
                indices.append(ind)
            cmp = num

        transl = tmp.iloc[indices]["auth_comp_id"].map(D3TO1).fillna("?")

        df = pd.concat((tmp.iloc[indices]["auth_asym_id"], transl), axis=1)
        df.columns = ["chain_id", "residue_name"]
        return df

    def get_model(self, model_index: int):
        """Return a new PandasPDB object with the dataframes subset to the given model index.

        Parameters
        ----------
        model_index : int
            An integer representing the model index to subset to.

        Returns
        -------
        pandas_pdb.PandasPdb : A new PandasPdb object containing the
          structure subsetted to the given model.

        """
        df = deepcopy(self)

        if "ATOM" in df.df.keys():
            df.df["ATOM"] = df.df["ATOM"].loc[
                df.df["ATOM"]["pdbx_PDB_model_num"] == model_index
            ]
        return df


def _retrieve_author_chain(chain):
    """Retrieve the (author) chain names present in the chain section (delimited by '|' chars) of a header line in a fasta file."""
    if "auth" in chain:
        return chain.split(" ")[-1][:-1]

    return chain


def _retrieve_chain_names(entry):
    """Retrieve the (author) chain names present in one header line of a fasta file (line that begins with '>')."""
    entry = entry.split("|")[1]

    if "Chains" in entry:
        return [_retrieve_author_chain(e) for e in entry[7:].split(", ")]

    return [_retrieve_author_chain(entry[6:])]


class _Atom(dict):
    """
    A class representing an atom in a PDB file (for visualization).

    Adapted from https://william-dawson.github.io/using-py3dmol.html
    """

    def __init__(self, row):
        self["type"] = row["record_name"]
        self["idx"] = row["atom_number"]
        self["name"] = row["atom_name"]
        self["resname"] = row["residue_name"]
        self["resid"] = row["residue_number"]
        self["chain"] = row["chain_id"]
        self["x"] = row["x_coord"]
        self["y"] = row["y_coord"]
        self["z"] = row["z_coord"]
        self["sym"] = row["element_symbol"]

    def __str__(self):
        line = list(" " * 80)

        line[0:6] = self["type"].ljust(6)
        line[6:11] = str(self["idx"]).ljust(5)
        line[12:16] = self["name"].ljust(4)
        line[17:20] = self["resname"].ljust(3)
        line[22:26] = str(self["resid"]).ljust(4)
        line[30:38] = str(self["x"]).rjust(8)
        line[38:46] = str(self["y"]).rjust(8)
        line[46:54] = str(self["z"]).rjust(8)
        line[76:78] = self["sym"].rjust(2)
        return "".join(line) + "\n"
