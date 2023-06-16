import os
import pickle
import warnings
from typing import Dict

import numpy as np
from Bio import pairwise2
from biopandas.pdb import PandasPdb
from einops import rearrange

from proteinflow.constants import (
    ALPHABET_PDB,
    ATOM_MAP_1,
    ATOM_MAP_3,
    ATOM_MAP_4,
    BACKBONE_ORDER,
    D3TO1,
    GLOBAL_PAD_CHAR,
    ONE_TO_THREE_LETTER_MAP,
    SIDECHAIN_ORDER,
)
from proteinflow.custom_mmcif import CustomMmcif
from proteinflow.sequences import (
    _compare_seqs,
    _get_chothia_cdr,
    _retrieve_fasta_chains,
)
from proteinflow.utils.common_utils import PDBError, _split_every


class PdbBuilder:
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
        seq = np.array([ALPHABET_PDB[x] for x in seq.int().numpy()])
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

    def _coord_generator(self):
        """Return a generator to iteratively yield self.atoms_per_res atoms at a time."""
        return self.coord_generator(self.coords, self.atoms_per_res)

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


def _open_pdb(file):
    """
    Open a PDB file in the pickle format that follows the dwnloading and processing of the database
    """

    with open(file, "rb") as f:
        return pickle.load(f)


def _align_structure(
    pdb_dict: Dict,
    min_length: int = 30,
    max_length: int = None,
    max_missing_middle: float = 0.1,
    max_missing_ends: float = 0.3,
    chain_id_string: str = None,
) -> Dict:
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
    - `'msk'`: a `numpy` array of shape `(L,)` where ones correspond to residues with known coordinates and
        zeros to missing values,
    - `'seq'`: a string of length `L` with residue types.

    If `chain_id_string` is provided, the output dictionary will also contain the following keys:
    - `'cdr'`: a `numpy` array of shape `(L,)` with CDR types (according to Chothia definition).

    Parameters
    ----------
    pdb_dict : Dict
        the output of `open_pdb`
    min_length : int, default 30
        the minimum number of non-missing residues per chain
    max_length : int, optional
        the maximum number of residues per chain
    chain_id_string: str, optional
        chain id string in the form of `"{pdb_id}_H_L_A1|...|An"` (for SAbDab)

    Returns
    -------
    pdb_dict : Dict | None
        the parsed dictionary or `None`, if the criteria are not met
    """

    crd = pdb_dict["crd_raw"]
    fasta = pdb_dict["fasta"]
    seq_df = pdb_dict["seq_df"]
    pdb_dict = {}
    crd = crd[crd["record_name"] == "ATOM"]

    if len(crd["chain_id"].unique()) == 0:
        raise PDBError("No chains found")

    if not crd["residue_name"].isin(D3TO1.keys()).all():
        raise PDBError("Unnatural amino acids found")

    chains_unique = crd["chain_id"].unique()
    chain_types = ["-" for _ in chains_unique]
    if chain_id_string is not None:
        H_chain, L_chain, A_chains = chain_id_string.split("_")
        A_chains = A_chains.split(" | ")
        if H_chain in ["NA", "nan"]:
            H_chain = None
        if L_chain in ["NA", "nan"]:
            L_chain = None
        if A_chains[0] in ["NA", "nan"]:
            A_chains = []
        if not set(A_chains).issubset(set(chains_unique)):
            raise PDBError("Antigen chains not found")
        if H_chain is not None and H_chain not in chains_unique:
            raise PDBError("Heavy chain not found")
        if L_chain is not None and L_chain not in chains_unique:
            raise PDBError("Light chain not found")
        chains_unique = []
        chain_types = []
        if H_chain is not None:
            chains_unique.append(H_chain)
            chain_types.append("H")
        if L_chain is not None:
            chains_unique.append(L_chain)
            chain_types.append("L")
        chains_unique.extend(A_chains)
        chain_types.extend(["-"] * len(A_chains))

    for chain, chain_type in zip(chains_unique, chain_types):
        pdb_dict[chain] = {}
        chain_crd = crd[crd["chain_id"] == chain].reset_index()

        if len(chain_crd) / len(fasta[chain]) < 1 - (
            max_missing_ends + max_missing_middle
        ):
            raise PDBError("Too many missing values in total")

        # align fasta and pdb and check criteria)
        pdb_seq = "".join(seq_df[seq_df["chain_id"] == chain]["residue_name"])
        if "insertion" in chain_crd.columns:
            chain_crd["residue_number"] = chain_crd.apply(
                lambda row: f"{row['residue_number']}_{row['insertion']}", axis=1
            )
        unique_numbers = chain_crd["residue_number"].unique()
        if len(unique_numbers) != len(pdb_seq):
            raise PDBError("Inconsistencies in the biopandas dataframe")
        # aligner = PairwiseAligner()
        # aligner.match_score = 2
        # aligner.mismatch_score = -10
        # aligner.open_gap_score = -0.5
        # aligner.extend_gap_score = -0.1
        # aligned_seq, fasta_seq = aligner.align(pdb_seq, fasta[chain])[0]
        aligned_seq, fasta_seq, *_ = pairwise2.align.globalms(
            pdb_seq, fasta[chain], 2, -10, -0.5, -0.1
        )[0]
        aligned_seq_arr = np.array(list(aligned_seq))
        if "-" in fasta_seq or "".join([x for x in aligned_seq if x != "-"]) != pdb_seq:
            raise PDBError("Incorrect alignment")
        residue_numbers = np.where(aligned_seq_arr != "-")[0]
        start = residue_numbers.min()
        end = residue_numbers.max() + 1
        if start + (len(aligned_seq) - end) > max_missing_ends * len(aligned_seq):
            raise PDBError("Too many missing values in the ends")
        if (aligned_seq_arr[start:end] == "-").sum() > max_missing_middle * (
            end - start
        ):
            raise PDBError("Too many missing values in the middle")
        if chain_id_string is not None:
            cdr_arr = np.array(["-"] * len(aligned_seq), dtype=object)
            if chain_type in ["H", "L"]:
                cdr_arr[aligned_seq_arr != "-"] = _get_chothia_cdr(
                    unique_numbers, chain_type
                )
            pdb_dict[chain]["cdr"] = cdr_arr
        pdb_dict[chain]["seq"] = fasta[chain]
        pdb_dict[chain]["msk"] = (aligned_seq_arr != "-").astype(int)
        known_length = sum(pdb_dict[chain]["msk"])
        if min_length is not None and known_length < min_length:
            raise PDBError("Sequence is too short")
        if max_length is not None and len(aligned_seq) > max_length:
            raise PDBError("Sequence is too long")

        # go over rows of coordinates
        crd_arr = np.zeros((len(aligned_seq), 14, 3))

        def arr_index(row):
            atom = row["atom_name"]
            if atom.startswith("H") or atom == "OXT":
                return -1  # ignore hydrogens and OXT
            order = BACKBONE_ORDER + SIDECHAIN_ORDER[row["residue_name"]]
            try:
                return order.index(atom)
            except ValueError:
                raise PDBError(f"Unexpected atoms ({atom})")

        indices = chain_crd.apply(arr_index, axis=1)
        indices = indices.astype(int)
        informative_mask = indices != -1
        res_indices = np.where(aligned_seq_arr != "-")[0]
        replace_dict = {x: y for x, y in zip(unique_numbers, res_indices)}
        chain_crd["residue_number"].replace(replace_dict, inplace=True)
        crd_arr[
            chain_crd[informative_mask]["residue_number"], indices[informative_mask]
        ] = chain_crd[informative_mask][["x_coord", "y_coord", "z_coord"]]

        pdb_dict[chain]["crd_bb"] = crd_arr[:, :4, :]
        pdb_dict[chain]["crd_sc"] = crd_arr[:, 4:, :]
        pdb_dict[chain]["msk"][(pdb_dict[chain]["crd_bb"] == 0).sum(-1).sum(-1) > 0] = 0
        if (pdb_dict[chain]["msk"][start:end] == 0).sum() > max_missing_middle * (
            end - start
        ):
            raise PDBError("Too many missing values in the middle")
    return pdb_dict


def _open_structure(
    file_path: str, tmp_folder: str, sabdab=False, chain_id=None
) -> Dict:
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
        the path to the .pdb.gz file
    tmp_folder : str
        the path to the temporary data folder
    sabdab : bool, default False
        whether the file is a SAbDab file
    chain_id : str, optional
        chain id string in the form of `"{pdb_id}_H_L_A1|...|An"` (for SAbDab)


    Output
    ------
    pdb_dict : Dict
        the parsed dictionary
    """

    cif = file_path.endswith("cif.gz")
    if sabdab:
        pdb = os.path.basename(file_path).split(".")[0]
    else:
        pdb, _ = os.path.basename(file_path).split(".")[0].split("-")
    out_dict = {}

    # download fasta and check if it contains only proteins
    fasta_path = os.path.join(tmp_folder, f"{pdb}.fasta")
    try:
        seqs_dict = _retrieve_fasta_chains(fasta_path)
    except FileNotFoundError:
        raise PDBError("Fasta file not found")

    # load coordinates in a nice format
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if cif:
                p = CustomMmcif().read_mmcif(file_path).get_model(1)
            else:
                p = PandasPdb().read_pdb(file_path).get_model(1)
    except FileNotFoundError:
        raise PDBError("PDB / mmCIF file downloaded but not found")
    out_dict["crd_raw"] = p.df["ATOM"]
    out_dict["seq_df"] = p.amino3to1()

    # retrieve sequences that are relevant for this PDB from the fasta file
    if chain_id is None:
        chains = p.df["ATOM"]["chain_id"].unique()
    else:
        h, l, a = chain_id.split("_")
        chains = [h, l] + a.split(" | ")
        chains = [x for x in chains if x != "nan"]
    seqs_dict = {k.upper(): v for k, v in seqs_dict.items()}
    if all([len(x) == 3 and len(set(list(x))) == 1 for x in seqs_dict.keys()]):
        seqs_dict = {k[0]: v for k, v in seqs_dict.items()}

    if not {x.split("-")[0].upper() for x in chains}.issubset(
        set(list(seqs_dict.keys()))
    ):
        raise PDBError("Some chains in the PDB do not appear in the fasta file")

    out_dict["fasta"] = {k: seqs_dict[k.split("-")[0].upper()] for k in chains}

    if not sabdab:
        try:
            os.remove(file_path)
        except OSError:
            pass
    return out_dict


def _check_biounits(biounits_list, threshold):
    """
    Return the indexes of the redundant biounits within the list of files given by `biounits_list`
    """

    biounits = [_open_pdb(b) for b in biounits_list]
    indexes = []

    for k, b1 in enumerate(biounits):
        if k not in indexes:
            b1_seqs = [b1[chain]["seq"] for chain in b1.keys()]
            for i, b2 in enumerate(biounits[k + 1 :]):
                if len(b1.keys()) != len(b2.keys()):
                    continue

                b2_seqs = [b2[chain]["seq"] for chain in b2.keys()]
                if _compare_seqs(b1_seqs, b2_seqs, threshold):
                    indexes.append(k + i + 1)

    return indexes
