"""
Adapted from `sidechainnet`.
"""

import itertools
import numpy as np
from proteinflow import ALPHABET
from einops import rearrange

GLOBAL_PAD_CHAR = 0
ONE_TO_THREE_LETTER_MAP = {
    "R": "ARG",
    "H": "HIS",
    "K": "LYS",
    "D": "ASP",
    "E": "GLU",
    "S": "SER",
    "T": "THR",
    "N": "ASN",
    "Q": "GLN",
    "C": "CYS",
    "G": "GLY",
    "P": "PRO",
    "A": "ALA",
    "V": "VAL",
    "I": "ILE",
    "L": "LEU",
    "M": "MET",
    "F": "PHE",
    "Y": "TYR",
    "W": "TRP",
}
ATOM_MAP_4 = {a: ["N", "C", "CA", "O"] for a in ONE_TO_THREE_LETTER_MAP.keys()}
ATOM_MAP_1 = {a: ["CA"] for a in ONE_TO_THREE_LETTER_MAP.keys()}
ATOM_MAP_3 = {a: ["N", "C", "CA"] for a in ONE_TO_THREE_LETTER_MAP.keys()}
ALPHABET = "XACDEFGHIKLMNPQRSTVWY"


def coord_generator(coords, atoms_per_res=14, remove_padding=False):
    """Return a generator to iteratively yield self.atoms_per_res atoms at a time."""
    coord_idx = 0
    while coord_idx < coords.shape[0]:
        _slice = coords[coord_idx : coord_idx + atoms_per_res]
        if remove_padding:
            non_pad_locs = (_slice != GLOBAL_PAD_CHAR).any(axis=1)
            _slice = _slice[non_pad_locs]
        yield _slice
        coord_idx += atoms_per_res


class PdbBuilder(object):
    """Creates a PDB file given a protein's atomic coordinates and sequence.

    The general idea is that if any model is capable of predicting a set of coordinates
    and mapping between those coordinates and residue/atom names, then this object can
    be use to transform that output into a PDB file.
    """

    def __init__(
        self,
        seq,
        coords,
        chain_dict,
        chain_id_arr,
        only_ca=False,
        skip_oxygens=False,
        mask=None,
    ):
        """
        Parameters
        ----------
        seq : torch.Tensor
            a 1D tensor of integers representing the sequence of the protein
        coords : torch.Tensor
            a 3D tensor of shape (L, 4, 3) where L is the number of atoms in the protein
            and the second dimension is the atom type (N, C, CA, O)
        chain_dict : dict
            a dictionary mapping chain IDs to chain names
        chain_id_arr : torch.Tensor
            a 1D tensor of integers representing the chain ID of each residue

        """
        seq = np.array([ALPHABET[x] for x in seq.int().numpy()])
        if only_ca:
            coords = coords[:, 2, :].unsqueeze(1)
        elif skip_oxygens:
            coords = coords[:, :-1, :]
        coords = rearrange(coords, "l n c -> (l n) c")

        if only_ca:
            atoms_per_res = 1
        elif skip_oxygens:
            atoms_per_res = 3
        else:
            atoms_per_res = 4
        terminal_atoms = None

        if len(seq) != coords.shape[0] / atoms_per_res:
            raise ValueError(
                "The sequence length must match the coordinate length and contain 1 "
                "letter AA codes."
                + str(coords.shape[0] / atoms_per_res)
                + " "
                + str(len(seq))
            )
        if coords.shape[0] % atoms_per_res != 0:
            raise AssertionError(
                f"Coords is not divisible by {atoms_per_res}. " f"{coords.shape}"
            )
        if atoms_per_res not in (
            4,
            1,
            3,
        ):
            raise ValueError("Invalid atoms_per_res. Must be 1, 3, or 4.")

        self.only_ca = only_ca
        self.skip_oxygens = skip_oxygens
        self.atoms_per_res = atoms_per_res
        self.has_hydrogens = False
        self.only_backbone = self.atoms_per_res == 4
        self.coords = coords
        self.seq = seq
        self.mapping = self._make_mapping_from_seq()
        self.terminal_atoms = terminal_atoms

        chain_dict_inv = {v: k for k, v in chain_dict.items()}
        self.chain_ids = np.array([chain_dict_inv[x] for x in chain_id_arr.numpy()])
        self.chain_ids_unique = list(chain_dict.keys())

        if mask is not None:
            self.chain_ids[mask.bool()] = "mask"
        self.chain_ids_unique.append("mask")

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
        return coord_generator(self.coords, self.atoms_per_res)

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

        # Add Terminal Atoms (Must be provided in terminal_atoms dict; Hs must be built)
        if n_terminal and self.terminal_atoms:
            residue_lines.append(
                self._get_line_for_atom(
                    res_name, "H2", self.terminal_atoms["H2"], chain_id
                )
            )
            residue_lines.append(
                self._get_line_for_atom(
                    res_name, "H3", self.terminal_atoms["H3"], chain_id
                )
            )
            self.atom_nbr += 2
        if c_terminal and self.terminal_atoms:
            residue_lines.append(
                self._get_line_for_atom(
                    res_name, "OXT", self.terminal_atoms["OXT"], chain_id
                )
            )

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
            # TODO assumes only first/last residue have terminal atoms
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
        """Given a protein sequence, this returns a mapping that assumes coords are
        generated in groups of atoms_per_res (the output is L x atoms_per_res x 3.)"""
        if self.only_ca:
            atom_names = ATOM_MAP_1
        elif self.skip_oxygens:
            atom_names = ATOM_MAP_3
        elif self.only_backbone:
            atom_names = ATOM_MAP_4
        else:
            raise NotImplementedError
        mapping = []
        for residue in self.seq:
            mapping.append((residue, atom_names[residue]))
        return mapping

    def get_pdb_string(self, title=None):
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
        seq = np.array(self.seq)
        lines = []
        for chain in self.chain_ids_unique:
            lineno = 1
            seq_chain = seq[self.chain_ids == chain]
            three_letter_seq = [ONE_TO_THREE_LETTER_MAP[c] for c in seq_chain]
            residue_blocks = list(split_every(13, three_letter_seq))
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

    def save_gltf(self, path, title="test", create_pdb=False):
        """First creates a PDB file, then converts it to GLTF and saves it to disk.

        Used for visualizing with Weights and Biases.
        """
        import pymol

        assert ".gltf" in path, "requested filepath must end with '.gtlf'."
        if create_pdb:
            self.save_pdb(path.replace(".gltf", ".pdb"), title)
        pymol.cmd.load(path.replace(".gltf", ".pdb"), title)
        pymol.cmd.color("oxygen", title)
        pymol.cmd.save(path, quiet=True)
        pymol.cmd.delete("all")


def split_every(n, iterable):
    """Split iterable into chunks. From https://stackoverflow.com/a/1915307/2780645."""
    i = iter(iterable)
    piece = list(itertools.islice(i, n))
    while piece:
        yield piece
        piece = list(itertools.islice(i, n))
