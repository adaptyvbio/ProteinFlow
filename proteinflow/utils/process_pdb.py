from Bio import pairwise2
import numpy as np
from typing import Dict
import subprocess
import urllib.request
import os
import numpy as np
from biopandas.pdb import PandasPdb
from proteinflow.utils.mmcif_fix import CustomMmcif
import os
from collections import namedtuple
from operator import attrgetter
import requests
import shutil
import warnings


SIDECHAIN_ORDER = {
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
BACKBONE_ORDER = ["N", "C", "CA", "O"]

d3to1 = {
    "CYS": "C",
    "ASP": "D",
    "SER": "S",
    "GLN": "Q",
    "LYS": "K",
    "ILE": "I",
    "PRO": "P",
    "THR": "T",
    "PHE": "F",
    "ASN": "N",
    "GLY": "G",
    "HIS": "H",
    "LEU": "L",
    "ARG": "R",
    "TRP": "W",
    "ALA": "A",
    "VAL": "V",
    "GLU": "E",
    "TYR": "Y",
    "MET": "M",
}
S3Obj = namedtuple("S3Obj", ["key", "mtime", "size", "ETag"])


class PDBError(ValueError):
    pass


def _s3list(
    bucket,
    path,
    start=None,
    end=None,
    recursive=True,
    list_dirs=True,
    list_objs=True,
    limit=None,
):
    """
    Iterator that lists a bucket's objects under path, (optionally) starting with
    start and ending before end.

    If recursive is False, then list only the "depth=0" items (dirs and objects).

    If recursive is True, then list recursively all objects (no dirs).

    Args:
        bucket:
            a boto3.resource('s3').Bucket().
        path:
            a directory in the bucket.
        start:
            optional: start key, inclusive (may be a relative path under path, or
            absolute in the bucket)
        end:
            optional: stop key, exclusive (may be a relative path under path, or
            absolute in the bucket)
        recursive:
            optional, default True. If True, lists only objects. If False, lists
            only depth 0 "directories" and objects.
        list_dirs:
            optional, default True. Has no effect in recursive listing. On
            non-recursive listing, if False, then directories are omitted.
        list_objs:
            optional, default True. If False, then directories are omitted.
        limit:
            optional. If specified, then lists at most this many items.

    Returns:
        an iterator of S3Obj.

    Examples:
        # set up
        >>> s3 = boto3.resource('s3')
        ... bucket = s3.Bucket('bucket-name')

        # iterate through all S3 objects under some dir
        >>> for p in s3list(bucket, 'some/dir'):
        ...     print(p)

        # iterate through up to 20 S3 objects under some dir, starting with foo_0010
        >>> for p in s3list(bucket, 'some/dir', limit=20, start='foo_0010'):
        ...     print(p)

        # non-recursive listing under some dir:
        >>> for p in s3list(bucket, 'some/dir', recursive=False):
        ...     print(p)

        # non-recursive listing under some dir, listing only dirs:
        >>> for p in s3list(bucket, 'some/dir', recursive=False, list_objs=False):
        ...     print(p)
    """

    kwargs = dict()
    if start is not None:
        if not start.startswith(path):
            start = os.path.join(path, start)
    if end is not None:
        if not end.startswith(path):
            end = os.path.join(path, end)
    if not recursive:
        kwargs.update(Delimiter="/")
        if not path.endswith("/") and len(path) > 0:
            path += "/"
    kwargs.update(Prefix=path)
    if limit is not None:
        kwargs.update(PaginationConfig={"MaxItems": limit})

    paginator = bucket.meta.client.get_paginator("list_objects")
    for resp in paginator.paginate(Bucket=bucket.name, **kwargs):
        q = []
        if "CommonPrefixes" in resp and list_dirs:
            q = [S3Obj(f["Prefix"], None, None, None) for f in resp["CommonPrefixes"]]
        if "Contents" in resp and list_objs:
            q += [
                S3Obj(f["Key"], f["LastModified"], f["Size"], f["ETag"])
                for f in resp["Contents"]
            ]
        # note: even with sorted lists, it is faster to sort(a+b)
        # than heapq.merge(a, b) at least up to 10K elements in each list
        q = sorted(q, key=attrgetter("key"))
        if limit is not None:
            q = q[:limit]
            limit -= len(q)
        for p in q:
            if end is not None and p.key >= end:
                return
            yield p


def _retrieve_author_chain(chain):
    """
    Retrieve the (author) chain names present in the chain section (delimited by '|' chars) of a header line in a fasta file
    """

    if "auth" in chain:
        return chain.split(" ")[-1][:-1]

    return chain


def _retrieve_chain_names(entry):
    """
    Retrieve the (author) chain names present in one header line of a fasta file (line that begins with '>')
    """

    entry = entry.split("|")[1]

    if "Chains" in entry:
        return [_retrieve_author_chain(e) for e in entry[7:].split(", ")]

    return [_retrieve_author_chain(entry[6:])]


def _retrieve_fasta_chains(fasta_file):
    """
    Return a dictionary containing all the (author) chains in a fasta file (keys) and their corresponding sequence
    """

    with open(fasta_file, "r") as f:
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


def _open_structure(file_path: str, tmp_folder: str) -> Dict:
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
    thr_resolution : float, default 3.5
        the resolution threshold

    Output
    ------
    pdb_dict : Dict
        the parsed dictionary
    """

    cif = file_path.endswith("cif.gz")
    pdb, biounit = os.path.basename(file_path).split(".")[0].split("-")
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

    # # add metadata
    # metadata = parse_pdb_header(file_path)
    # for key in ["structure_method"]:
    #     out_dict[key] = metadata.get(key)

    # retrieve sequences that are relevant for this PDB from the fasta file
    chains = p.df["ATOM"]["chain_id"].unique()

    if not set([x.split("-")[0] for x in chains]).issubset(set(list(seqs_dict.keys()))):
        raise PDBError("Some chains in the PDB do not appear in the fasta file")

    out_dict["fasta"] = {k: seqs_dict[k.split("-")[0]] for k in chains}

    try:
        os.remove(file_path)
    except OSError:
        pass
    return out_dict


def _align_structure(
    pdb_dict: Dict,
    min_length: int = 30,
    max_length: int = None,
    max_missing_middle: float = 0.1,
    max_missing_ends: float = 0.3,
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
    seq_df = pdb_dict["seq_df"]
    pdb_dict = {}
    crd = crd[crd["record_name"] == "ATOM"]

    if len(crd["chain_id"].unique()) == 0:
        raise PDBError("No chains found")

    if not crd["residue_name"].isin(d3to1.keys()).all():
        raise PDBError("Unnatural amino acids found")

    for chain in crd["chain_id"].unique():
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
        pdb_dict[chain]["seq"] = fasta[chain]
        pdb_dict[chain]["msk"] = (aligned_seq_arr != "-").astype(int)
        l = sum(pdb_dict[chain]["msk"])
        if min_length is not None and l < min_length:
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
            except:
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
