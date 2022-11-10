from Bio import pairwise2
import numpy as np
from typing import Dict
import subprocess
import gzip
import shutil
import urllib.request
import os
import re
import numpy as np
import pickle as pkl
from Bio.Seq import Seq
from Bio.PDB import PDBParser
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.PDBIO import Select
from Bio.PDB.parse_pdb_header import parse_pdb_header
from biopandas.pdb import PandasPdb
import os
import boto3
from collections import namedtuple
from operator import attrgetter


S3Obj = namedtuple("S3Obj", ["key", "mtime", "size", "ETag"])


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

bb_names = ["N", "C", "CA", "O"]


class PDBError(ValueError):
    pass


class SelectHeavyAtoms(Select):
    def accept_residue(self, residue):
        return residue.id[0] == " "

    def accet_atom(self, atom):
        return atom.id[0] != "H"


def s3list(
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
        # note: need to use a string just smaller than start, because
        # the list_object API specifies that start is excluded (the first
        # result is *after* start).
        kwargs.update(Marker=__prev_str(start))
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


def get_pdb_file(pdb_file, bucket, tmp_folder, folders):
    """
    Download the file from S3 and return the local path where it was saved
    """

    id = os.path.basename(pdb_file)
    pdb_id = id.split(".")[0]
    biounit = id.split(".")[1][3]
    local_path = os.path.join(tmp_folder, f"{pdb_id}-{biounit}.pdb.gz")
    for folder in folders: # go over database snapshots in case the file is not found in the latest one
        file = folder + pdb_file
        try:
            bucket.download_file(file, local_path)
            return local_path
        except:
            continue
    raise PDBError(f"Could not download")


def download_fasta(pdbcode, biounit, datadir):
    """
    Downloads a fasta file from the Internet and saves it in a data directory.
    For informations about the download url, cf `https://www.rcsb.org/pages/download/http#structures`
    :param pdbcode: The standard PDB ID e.g. '3ICB' or '3icb'
    :param datadir: The directory where the downloaded file will be saved
    :return: the full path to the downloaded PDB file or None if something went wrong
    """

    downloadurl = "https://www.rcsb.org/fasta/entry/"
    pdbfn = pdbcode + "/download"
    outfnm = os.path.join(datadir, f"{pdbcode}-{biounit}.fasta")

    url = downloadurl + pdbfn
    try:
        urllib.request.urlretrieve(url, outfnm)
        return outfnm

    except Exception as err:
        # print(str(err), file=sys.stderr)
        return None, err


def validate(seq, alphabet="dna"):

    """
    Check that a given sequence contains only proteic residues
    """

    alphabets = {
        "dna": re.compile("^[acgtn]*$", re.I),
        "protein": re.compile("^[acdefghiklmnpqrstvwy]*$", re.I),
    }

    return alphabets[alphabet].search(seq) is not None


def detect_non_proteins(fasta_file):

    """
    Detect if a fasta contains residues that do not belong to a protein (DNA, RNA, non-canonical amino acids, ...)
    """

    with open(fasta_file, "r") as f:

        seq = Seq(
            "".join(
                [line.replace("\n", "") for line in f.readlines() if line[0] != ">"]
            )
        )

    return validate(str(seq), "dna") or not validate(str(seq), "protein")


def retrieve_author_chain(chain):

    """
    Retrieve the (author) chain names present in the chain section (delimited by '|' chars) of a header line in a fasta file
    """

    if "auth" in chain:
        return chain.split(" ")[-1][:-1]

    return chain


def retrieve_chain_names(entry):

    """
    Retrieve the (author) chain names present in one header line of a fasta file (line that begins with '>')
    """

    entry = entry.split("|")[1]

    if "Chains" in entry:
        return [retrieve_author_chain(e) for e in entry[7:].split(", ")]

    return [retrieve_author_chain(entry[6:])]


def retrieve_fasta_chains(fasta_file):

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
        for chain in retrieve_chain_names(name):
            out_dict[chain] = seq

    return out_dict


# def retrieve_pdb_resolution(pdb_id, tmp_folder, bucket):

#     """
#     Find the resolution of the PDB by downloading the PDB from the web
#     """

#     pdb_file = 'pdb' + pdb_id + '.ent.gz'
#     pdb_unzipped = os.path.join(tmp_folder, pdb_id + '_full' + '.pdb')
#     download_path = '20220103/pub/pdb/data/structures/all/pdb/' + pdb_file
#     local_path = get_pdb_file(download_path, bucket=bucket, tmp_folder=tmp_folder)

#     with gzip.open(local_path, 'rb') as f_in:
#         with open(pdb_unzipped, 'wb') as f_out:
#             shutil.copyfileobj(f_in, f_out)

#     os.remove(local_path)
#     header = parse_pdb_header(pdb_unzipped)
#     os.remove(pdb_unzipped)
#     return header['resolution']


def check_resolution(pdb_id, resolution_dict, tmp_folder, bucket):

    """
    Find the resolution of the PDB by first checking into the resolution dictionary and then by downloading the PDB from the web if necessary
    """

    with open(resolution_dict, "rb") as f:
        res_dict = pkl.load(f)

    if pdb_id in res_dict.keys():
        resolution = res_dict[pdb_id]

    else:
        resolution = retrieve_pdb_resolution(pdb_id, tmp_folder, bucket)
        res_dict[pdb_id] = resolution
        with open(resolution_dict, "wb") as f:
            pkl.dump(res_dict, f)

    return resolution


def open_pdb(file_path: str, tmp_folder: str) -> Dict:
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
        the path to the .pdb{i}.gz file (i is a positive integer)
    tmp_folder : str
        the path to the temporary data folder
    thr_resolution : float, default 3.5
        the resolution threshold

    Output
    ------
    pdb_dict : Dict
        the parsed dictionary

    """

    pdb, biounit = os.path.basename(file_path).split("-")
    out_dict = {}

    # download fasta and check if it contains only proteins
    fasta_path = download_fasta(pdb, biounit, tmp_folder)
    try:
        seqs_dict = retrieve_fasta_chains(fasta_path)
    except FileNotFoundError:
        raise PDBError("Fasta file not found")

    # load coordinates in a nice format
    try:
        p = PandasPdb().read_pdb(file_path).df["ATOM"]
    except FileNotFoundError:
        raise PDBError("PDB file not found")
    out_dict["crd_raw"] = p

    # retrieve sequences that are relevant for this PDB from the fasta file
    chains = np.unique(p["chain_id"].values)

    if not set(chains).issubset(set(list(seqs_dict.keys()))):
        raise PDBError("Some chains in the PDB do not appear in the fasta file")

    out_dict["fasta"] = {k: seqs_dict[k] for k in chains}

    for path in [file_path, fasta_path]:
        if os.path.exists(path):
            subprocess.run(
                ["rm", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

    return out_dict


def align_pdb(
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
    pdb_dict = {}
    crd = crd[crd["record_name"] == "ATOM"]

    if not crd["residue_name"].isin(d3to1.keys()).all():
        raise PDBError("Unnatural amino acids found")

    for chain in crd["chain_id"].unique():
        pdb_dict[chain] = {}
        chain_crd = crd[crd["chain_id"] == chain].reset_index()
        atom_numbers = list(chain_crd["atom_number"])

        # check for multiple models
        if atom_numbers[0] in atom_numbers[1:]:
            index_1 = atom_numbers[1:].index(atom_numbers[0]) + 1
            chain_crd = chain_crd.iloc[:index_1]

        # align fasta and pdb and check criteria
        indices = np.where(
            np.diff(np.pad(chain_crd["residue_number"], (1, 0), constant_values=-10))
            != 0
        )
        pdb_seq = "".join([d3to1[x] for x in chain_crd.loc[indices]["residue_name"]])
        if len(pdb_seq) / len(fasta[chain]) < 1 - (
            max_missing_ends + max_missing_middle
        ):
            raise PDBError("Too many missing values in total")
        aligned_seq, fasta_seq, *_ = pairwise2.align.globalms(
            pdb_seq, fasta[chain], 2, -10, -0.5, -0.1
        )[0]
        if "-" in fasta_seq:
            raise PDBError("Incorrect alignment")
        if "".join([x for x in aligned_seq if x != "-"]) != pdb_seq:
            raise PDBError("Incorrect alignment")
        start = 0
        while aligned_seq[start] == "-":
            start += 1
        end = len(aligned_seq)
        while aligned_seq[end - 1] == "-":
            end -= 1
        if start + (len(aligned_seq) - end) > max_missing_ends * len(aligned_seq):
            raise PDBError("Too many missing values in the ends")
        if sum(
            [aligned_seq[i] == "-" for i in range(start, end)]
        ) > max_missing_middle * (end - start):
            raise PDBError("Too many missing values in the middle")
        pdb_dict[chain]["seq"] = fasta_seq
        pdb_dict[chain]["msk"] = (np.array(list(aligned_seq)) != "-").astype(int)
        l = sum(pdb_dict[chain]["msk"])
        if l < min_length:
            raise PDBError("Sequence is too short")
        if max_length is not None and len(aligned_seq) > max_length:
            raise PDBError("Sequence is too long")

        # go over rows of coordinates
        crd_arr = np.zeros((len(aligned_seq), 14, 3))
        seq_pos = -1
        pdb_pos = None
        for row_i, row in chain_crd.iterrows():
            res_num = row["residue_number"]
            res_name = row["residue_name"]
            atom = row["atom_name"]
            if res_num != pdb_pos:
                seq_pos += 1
                pdb_pos = res_num
                while aligned_seq[seq_pos] == "-" and seq_pos < len(aligned_seq) - 1:
                    seq_pos += 1
                if d3to1[res_name] != aligned_seq[seq_pos]:
                    raise PDBError("Alignment issue in processing")
            if atom not in bb_names + side_chain[res_name]:
                if atom in ["OXT"] or atom.startswith("H"):  # extra oxygen or hydrogen
                    continue
                raise PDBError(f"Unexpected atoms ({atom})")
            else:
                crd_arr[
                    seq_pos, (bb_names + side_chain[res_name]).index(atom), :
                ] = row[["x_coord", "y_coord", "z_coord"]]
        pdb_dict[chain]["crd_bb"] = crd_arr[:, :4, :]
        pdb_dict[chain]["crd_sc"] = crd_arr[:, 4:, :]
        pdb_dict[chain]["msk"][(pdb_dict[chain]["crd_bb"] == 0).sum(-1).sum(-1) > 0] = 0
        if (pdb_dict[chain]["msk"][start: end] == 0).sum() > max_missing_middle * (end - start):
            raise PDBError("Too many missing values in the middle (backbone atoms")
    return pdb_dict
