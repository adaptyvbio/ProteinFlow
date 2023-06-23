import os
import pickle
import random
import shutil
import subprocess
from collections import Counter, defaultdict
from copy import deepcopy
from itertools import combinations

import numpy as np
import pandas as pd
import torch
from numpy import linalg
from p_tqdm import p_map
from torch.utils.data import Dataset
from tqdm import tqdm

from proteinflow.constants import _PMAP, ALPHABET, CDR, D3TO1, MAIN_ATOMS
from proteinflow.pdb import _check_biounits
from proteinflow.utils.biotite_sse import _annotate_sse
from proteinflow.utils.boto_utils import (
    _download_dataset_dicts_from_s3,
    _download_dataset_from_s3,
    _get_s3_paths_from_tag,
)


class ProteinDataset(Dataset):
    """
    Dataset to load proteinflow data

    Saves the model input tensors as pickle files in `features_folder`. When `clustering_dict_path` is provided,
    at each iteration a random biounit from a cluster is sampled.

    If a complex contains multiple chains, they are concatenated. The sequence identity information is preserved in the
    `'chain_encoding_all'` object and in the `'residue_idx'` arrays the chain change is denoted by a +100 jump.

    Returns dictionaries with the following keys and values (all values are `torch` tensors):

    - `'X'`: 3D coordinates of N, C, Ca, O, `(total_L, 4, 3)`,
    - `'S'`: sequence indices (shape `(total_L)`),
    - `'mask'`: residue mask (0 where coordinates are missing, 1 otherwise; with interpolation 0s are replaced with 1s), `(total_L)`,
    - `'mask_original'`: residue mask (0 where coordinates are missing, 1 otherwise; not changed with interpolation), `(total_L)`,
    - `'residue_idx'`: residue indices (from 0 to length of sequence, +100 where chains change), `(total_L)`,
    - `'chain_encoding_all'`: chain indices, `(total_L)`,
    - `'chain_id`': a sampled chain index,
    - `'chain_dict'`: a dictionary of chain ids (keys are chain ids, e.g. `'A'`, values are the indices used in `'chain_id'` and `'chain_encoding_all'` objects)

    You can also choose to include additional features (set in the `node_features_type` parameter):

    - `'sidechain_orientation'`: a unit vector in the direction of the sidechain, `(total_L, 3)`,
    - `'dihedral'`: the dihedral angles, `(total_L, 2)`,
    - `'chemical'`: hydropathy, volume, charge, polarity, acceptor/donor features, `(total_L, 6)`,
    - `'secondary_structure'`: a one-hot encoding of secondary structure ([alpha-helix, beta-sheet, coil]), `(total_L, 3)`,
    - `'sidechain_coords'`: the coordinates of the sidechain atoms (see `proteinflow.sidechain_order()` for the order), `(total_L, 10, 3)`.

    If the dataset contains a `'cdr'` key (if it was generated from SAbDab files), the output files will also additionally contain a `'cdr'`
    key with a CDR tensor of length `total_L`. In the array, the CDR residues are marked with the corresponding CDR type
    (H1=1, H2=2, H3=3, L1=4, L2=5, L3=6) and the rest of the residues are marked with 0s.

    Use the `set_cdr` method to only iterate over specific CDRs.

    In order to compute additional features, use the `feature_functions` parameter. It should be a dictionary with keys
    corresponding to the feature names and values corresponding to the functions that compute the features. The functions
    should take a chain dictionary and an integer representation of the sequence as input (the dictionary is in `proteinflow` format,
    see the docs for `generate_data` for details) and return a `numpy` array shaped as `(#residues, #features)`.

    """

    def __init__(
        self,
        dataset_folder,
        features_folder="./data/tmp/",
        clustering_dict_path=None,
        max_length=None,
        rewrite=False,
        use_fraction=1,
        load_to_ram=False,
        debug=False,
        interpolate="none",
        node_features_type="zeros",
        debug_file_path=None,
        entry_type="biounit",  # biounit, chain, pair
        classes_to_exclude=None,  # heteromers, homomers, single_chains
        shuffle_clusters=True,
        min_cdr_length=None,
        feature_functions=None,
        classes_dict_path=None,
        cut_edges=False,
    ):
        """
        Parameters
        ----------
        dataset_folder : str
            the path to the folder with proteinflow format input files (assumes that files are named {biounit_id}.pickle)
        features_folder : str, default "./data/tmp/"
            the path to the folder where the ProteinMPNN features will be saved
        clustering_dict_path : str, optional
            path to the pickled clustering dictionary (keys are cluster ids, values are (biounit id, chain id) tuples)
        max_length : int, optional
            entries with total length of chains larger than `max_length` will be disregarded
        rewrite : bool, default False
            if `False`, existing feature files are not overwritten
        use_fraction : float, default 1
            the fraction of the clusters to use (first N in alphabetic order)
        load_to_ram : bool, default False
            if `True`, the data will be stored in RAM (use with caution! if RAM isn't big enough the machine might crash)
        debug : bool, default False
            only process 1000 files
        interpolate : {"none", "only_middle", "all"}
            `"none"` for no interpolation, `"only_middle"` for only linear interpolation in the middle, `"all"` for linear interpolation + ends generation
        node_features_type : {"zeros", "dihedral", "sidechain_orientation", "chemical", "secondary_structure" or combinations with "+"}
            the type of node features, e.g. `"dihedral"` or `"sidechain_orientation+chemical"`
        debug_file_path : str, optional
            if not `None`, open this single file instead of loading the dataset
        entry_type : {"biounit", "chain", "pair"}
            the type of entries to generate (`"biounit"` for biounit-level complexes, `"chain"` for chain-level, `"pair"`
            for chain-chain pairs (all pairs that are seen in the same biounit and have intersecting coordinate clouds))
        classes_to_exclude : list of str, optional
            a list of classes to exclude from the dataset (select from `"single_chains"`, `"heteromers"`, `"homomers"`)
        shuffle_clusters : bool, default True
            if `True`, a new representative is randomly selected for each cluster at each epoch (if `clustering_dict_path` is given)
        min_cdr_length : int, optional
            for SAbDab datasets, biounits with CDRs shorter than `min_cdr_length` will be excluded
        feature_functions : dict, optional
            a dictionary of functions to compute additional features (keys are the names of the features, values are the functions)
        classes_dict_path : str, optional
            a path to a pickled dictionary with biounit classes (single chain / heteromer / homomer)
        cut_edges : bool, default False
            if `True`, missing values at the edges of the sequence will be cut off

        """
        alphabet = ALPHABET
        self.alphabet_dict = defaultdict(lambda: 0)
        for i, letter in enumerate(alphabet):
            self.alphabet_dict[letter] = i
        self.alphabet_dict["X"] = 0
        self.files = defaultdict(lambda: defaultdict(list))  # file path by biounit id
        self.loaded = None
        self.dataset_folder = dataset_folder
        self.features_folder = features_folder
        self.cut_edges = cut_edges
        self.feature_types = []
        if node_features_type is not None:
            self.feature_types = node_features_type.split("+")
        self.entry_type = entry_type
        self.shuffle_clusters = shuffle_clusters
        self.feature_functions = {
            "sidechain_orientation": self._sidechain,
            "dihedral": self._dihedral,
            "chemical": self._chemical,
            "secondary_structure": self._sse,
            "sidechain_coords": self._sidechain_coords,
        }
        self.feature_functions.update(feature_functions or {})
        if classes_to_exclude is not None and not all(
            [
                x in ["single_chains", "heteromers", "homomers"]
                for x in classes_to_exclude
            ]
        ):
            raise ValueError(
                "Invalid class to exclude, choose from 'single_chains', 'heteromers', 'homomers'"
            )

        if debug_file_path is not None:
            self.dataset_folder = os.path.dirname(debug_file_path)
            debug_file_path = os.path.basename(debug_file_path)

        self.main_atom_dict = defaultdict(lambda: None)
        d1to3 = {v: k for k, v in D3TO1.items()}
        for i, letter in enumerate(alphabet):
            if i == 0:
                continue
            self.main_atom_dict[i] = MAIN_ATOMS[d1to3[letter]]

        # create feature folder if it does not exist
        if not os.path.exists(self.features_folder):
            os.makedirs(self.features_folder)

        self.interpolate = interpolate
        # generate the feature files
        print("Processing files...")
        if debug_file_path is None:
            to_process = [
                x for x in os.listdir(dataset_folder) if x.endswith(".pickle")
            ]
        else:
            to_process = [debug_file_path]
        if clustering_dict_path is not None and use_fraction < 1:
            with open(clustering_dict_path, "rb") as f:
                clusters = pickle.load(f)
            keys = sorted(clusters.keys())[: int(len(clusters) * use_fraction)]
            to_process = set()
            for key in keys:
                to_process.update([x[0] for x in clusters[key]])

            file_set = set(os.listdir(dataset_folder))
            to_process = [x for x in to_process if x in file_set]
        if debug:
            to_process = to_process[:1000]
        if self.entry_type == "pair":
            print(
                "Please note that the pair entry type takes longer to process than the other two. The progress bar is not linear because of the varying number of chains per file."
            )
        output_tuples_list = p_map(
            lambda x: self._process(
                x, rewrite=rewrite, max_length=max_length, min_cdr_length=min_cdr_length
            ),
            to_process,
        )
        # save the file names
        for output_tuples in output_tuples_list:
            for id, filename, chain_set in output_tuples:
                for chain in chain_set:
                    self.files[id][chain].append(filename)
        if classes_to_exclude is None:
            classes_to_exclude = []
        elif classes_dict_path is None:
            raise ValueError(
                "classes_to_exclude is not None, but classes_dict_path is None"
            )
        if clustering_dict_path is not None:
            if entry_type == "pair":
                classes_to_exclude = set(classes_to_exclude)
                classes_to_exclude.add("single_chains")
                classes_to_exclude = list(classes_to_exclude)
            with open(clustering_dict_path, "rb") as f:
                self.clusters = pickle.load(f)  # list of biounit ids by cluster id
                try:  # old way of storing class information
                    classes = pickle.load(f)
                except EOFError:
                    with open(classes_dict_path, "rb") as f:
                        classes = pickle.load(f)
            to_exclude = set()
            for c in classes_to_exclude:
                for key, id_arr in classes.get(c, {}).items():
                    for id, _ in id_arr:
                        to_exclude.add(id)
            for key in list(self.clusters.keys()):
                cluster_list = []
                for x in self.clusters[key]:
                    if x[0] in to_exclude:
                        continue
                    id = x[0].split(".")[0]
                    chain = x[1]
                    if id not in self.files:
                        continue
                    if chain not in self.files[id]:
                        continue
                    if len(self.files[id][chain]) == 0:
                        continue
                    cluster_list.append([id, chain])
                self.clusters[key] = cluster_list
                if len(self.clusters[key]) == 0:
                    self.clusters.pop(key)
            self.data = list(self.clusters.keys())
        else:
            self.clusters = None
            self.data = list(self.files.keys())
        # create a smaller dataset if necessary (if we have clustering it's applied earlier)
        if clustering_dict_path is None and use_fraction < 1:
            self.data = sorted(self.data)[: int(len(self.data) * use_fraction)]
        if load_to_ram:
            print("Loading to RAM...")
            self.loaded = {}
            seen = set()
            for id in self.files:
                for chain, file_list in self.files[id].items():
                    for file in file_list:
                        if file in seen:
                            continue
                        seen.add(file)
                        with open(file, "rb") as f:
                            self.loaded[file] = pickle.load(f)
        sample_file = list(self.files.keys())[0]
        sample_chain = list(self.files[sample_file].keys())[0]
        self.sabdab = "__" in sample_chain
        self.cdr = 0
        self.set_cdr(None)

    def _interpolate(self, crd_i, mask_i):
        """
        Fill in missing values in the middle with linear interpolation and (if fill_ends is true) build an initialization for the ends

        For the ends, the first 10 residues are 3.6 A apart from each other on a straight line from the last known value away from the center.
        Next they are 3.6 A apart in a random direction.
        """

        if self.interpolate in ["all", "only_middle"]:
            crd_i[(1 - mask_i).astype(bool)] = np.nan
            df = pd.DataFrame(crd_i.reshape((crd_i.shape[0], -1)))
            crd_i = df.interpolate(limit_area="inside").values.reshape(crd_i.shape)
        if self.interpolate == "all":
            non_nans = np.where(~np.isnan(crd_i[:, 0, 0]))[0]
            known_start = non_nans[0]
            known_end = non_nans[-1] + 1
            if known_end < len(crd_i) or known_start > 0:
                center = crd_i[non_nans, 2, :].mean(0)
                if known_start > 0:
                    direction = crd_i[known_start, 2, :] - center
                    direction = direction / linalg.norm(direction)
                    for i in range(0, min(known_start, 10)):
                        crd_i[known_start - i - 1] = (
                            crd_i[known_start - i] + direction * 3.6
                        )
                    for i in range(min(known_start, 10), known_start):
                        v = np.random.rand(3)
                        v = v / linalg.norm(v)
                        crd_i[known_start - i - 1] = crd_i[known_start - i] + v * 3.6
                if known_end < len(crd_i):
                    to_add = len(crd_i) - known_end
                    direction = crd_i[known_end - 1, 2, :] - center
                    direction = direction / linalg.norm(direction)
                    for i in range(0, min(to_add, 10)):
                        crd_i[known_end + i] = (
                            crd_i[known_end + i - 1] + direction * 3.6
                        )
                    for i in range(min(to_add, 10), to_add):
                        v = np.random.rand(3)
                        v = v / linalg.norm(v)
                        crd_i[known_end + i] = crd_i[known_end + i - 1] + v * 3.6
            mask_i = np.ones(mask_i.shape)
        if self.interpolate in ["only_middle"]:
            nan_mask = np.isnan(crd_i)  # in the middle the nans have been interpolated
            mask_i[~np.isnan(crd_i[:, 0, 0])] = 1
            crd_i[nan_mask] = 0
        if self.interpolate == "zeros":
            non_nans = np.where(mask_i != 0)[0]
            known_start = non_nans[0]
            known_end = non_nans[-1] + 1
            mask_i[known_start:known_end] = 1
        return crd_i, mask_i

    def _dihedral_angle(self, crd, msk):
        """Praxeolitic formula
        1 sqrt, 1 cross product"""

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

    def _dihedral(self, chain_dict, seq):
        """
        Dihedral angles
        """

        crd = chain_dict["crd_bb"]
        msk = chain_dict["msk"]
        angles = []
        # N, C, Ca, O
        # psi
        p = crd[:-1, [0, 2, 1], :]
        p = np.concatenate([p, crd[1:, [0], :]], 1)
        p = np.pad(p, ((0, 1), (0, 0), (0, 0)))
        angles.append(self._dihedral_angle(p, msk))
        # phi
        p = crd[:-1, [1], :]
        p = np.concatenate([p, crd[1:, [0, 2, 1]]], 1)
        p = np.pad(p, ((1, 0), (0, 0), (0, 0)))
        angles.append(self._dihedral_angle(p, msk))
        angles = np.stack(angles, -1)
        return angles

    def _sidechain(self, chain_dict, seq):
        """
        Sidechain orientation (defined by the 'main atoms' in the `main_atom_dict` dictionary)
        """

        crd_sc = chain_dict["crd_sc"]
        crd_bb = chain_dict["crd_bb"]
        orientation = np.zeros((crd_sc.shape[0], 3))
        for i in range(1, 21):
            if self.main_atom_dict[i] is not None:
                orientation[seq == i] = (
                    crd_sc[seq == i, self.main_atom_dict[i], :] - crd_bb[seq == i, 2, :]
                )
            else:
                S_mask = seq == i
                orientation[S_mask] = np.random.rand(*orientation[S_mask].shape)
        orientation /= np.expand_dims(linalg.norm(orientation, axis=-1), -1) + 1e-7
        return orientation

    def _chemical(self, chain_dict, seq):
        """
        Chemical features (hydropathy, volume, charge, polarity, acceptor/donor)
        """

        features = np.array([_PMAP(x) for x in seq])
        return features

    def _sse(self, chain_dict, seq):
        """
        Secondary structure features
        """

        sse_map = {"c": [0, 0, 1], "b": [0, 1, 0], "a": [1, 0, 0], "": [0, 0, 0]}
        sse = _annotate_sse(chain_dict["crd_bb"])
        sse = np.array([sse_map[x] for x in sse]) * chain_dict["msk"][:, None]
        return sse

    def _sidechain_coords(self, chain_dict, seq):
        """
        Sidechain coordinates
        """

        crd_sc = chain_dict["crd_sc"]
        return crd_sc

    def _process(self, filename, rewrite=False, max_length=None, min_cdr_length=None):
        """
        Process a proteinflow file and save it as ProteinMPNN features
        """

        input_file = os.path.join(self.dataset_folder, filename)
        no_extension_name = filename.split(".")[0]
        with open(input_file, "rb") as f:
            data = pickle.load(f)
        chains = sorted(data.keys())
        if self.entry_type == "biounit":
            chain_sets = [chains]
        elif self.entry_type == "chain":
            chain_sets = [[x] for x in chains]
        elif self.entry_type == "pair":
            chain_sets = list(combinations(chains, 2))
        else:
            raise RuntimeError(
                "Unknown entry type, please choose from ['biounit', 'chain', 'pair']"
            )
        output_names = []
        for chains_i, chain_set in enumerate(chain_sets):
            output_file = os.path.join(
                self.features_folder, no_extension_name + f"_{chains_i}.pickle"
            )
            pass_set = False
            add_name = True
            if os.path.exists(output_file) and not rewrite:
                pass_set = True
                if max_length is not None:
                    if sum([len(data[x]["seq"]) for x in chain_set]) > max_length:
                        add_name = False
                if min_cdr_length is not None:
                    for chain in chain_set:
                        if "cdr" not in data[chain]:
                            continue
                        u = np.unique(data[chain]["cdr"])
                        for cdr_ in u:
                            if (data[chain]["cdr"] == cdr_).sum() < min_cdr_length:
                                add_name = False
            else:
                X = []
                S = []
                mask = []
                mask_original = []
                chain_encoding_all = []
                residue_idx = []
                cdr = []
                node_features = defaultdict(lambda: [])
                last_idx = 0
                chain_dict = {}

                if max_length is not None:
                    if sum([len(data[x]["seq"]) for x in chain_set]) > max_length:
                        pass_set = True
                        add_name = False
                if min_cdr_length is not None:
                    for chain in chain_set:
                        if "cdr" not in data[chain]:
                            continue
                        u = np.unique(data[chain]["cdr"])
                        for cdr_ in u:
                            if (data[chain]["cdr"] == cdr_).sum() < min_cdr_length:
                                add_name = False
                                pass_set = True

                if self.entry_type == "pair":
                    # intersect = []
                    X1 = data[chain_set[0]]["crd_bb"][
                        data[chain_set[0]]["msk"].astype(bool)
                    ]
                    X2 = data[chain_set[1]]["crd_bb"][
                        data[chain_set[1]]["msk"].astype(bool)
                    ]
                    intersect_dim_X1 = []
                    intersect_dim_X2 = []
                    intersect_X1 = np.zeros(len(X1))
                    intersect_X2 = np.zeros(len(X2))
                    margin = 30
                    cutoff = 10
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

                        # if min_dim_1 - 4 <= max_dim_2 and max_dim_1 >= min_dim_2 - 4:
                        #     intersect.append(True)
                        # else:
                        #     intersect.append(False)
                        #     break
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

                    # distances = torch.norm(X1[intersect_X1, 2, :] - X2[intersect_X2, 2, :](1), dim=-1)
                    diff = X1[intersect_X1, 2, np.newaxis, :] - X2[intersect_X2, 2, :]
                    distances = np.sqrt(np.sum(diff**2, axis=2))

                    intersect_X1 = torch.LongTensor(intersect_X1)
                    intersect_X2 = torch.LongTensor(intersect_X2)
                    if np.sum(distances < cutoff) < 3:
                        # if not all(intersect):
                        pass_set = True
                        add_name = False
            if pass_set:
                continue

            cdr_chain_set = set()
            for chain_i, chain in enumerate(chain_set):
                seq = torch.tensor([self.alphabet_dict[x] for x in data[chain]["seq"]])
                S.append(seq)
                mask_original.append(deepcopy(data[chain]["msk"]))
                if self.interpolate != "none":
                    data[chain]["crd_bb"], data[chain]["msk"] = self._interpolate(
                        data[chain]["crd_bb"], data[chain]["msk"]
                    )
                X.append(data[chain]["crd_bb"])
                mask.append(data[chain]["msk"])
                residue_idx.append(torch.arange(len(data[chain]["seq"])) + last_idx)
                if "cdr" in data[chain]:
                    u, inv = np.unique(data[chain]["cdr"], return_inverse=True)
                    cdr_chain = np.array([CDR[x] for x in u])[inv].reshape(
                        data[chain]["cdr"].shape
                    )
                    cdr.append(cdr_chain)
                    cdr_chain_set.update([f"{chain}__{cdr}" for cdr in u])
                last_idx = residue_idx[-1][-1] + 100
                chain_encoding_all.append(torch.ones(len(data[chain]["seq"])) * chain_i)
                chain_dict[chain] = chain_i
                for name in self.feature_types:
                    if name not in self.feature_functions:
                        continue
                    func = self.feature_functions[name]
                    node_features[name].append(func(data[chain], seq))

            if add_name:
                output_names.append(
                    (
                        os.path.basename(no_extension_name),
                        output_file,
                        chain_set if len(cdr_chain_set) == 0 else cdr_chain_set,
                    )
                )

            if self.cut_edges:
                for chain_i, m in enumerate(mask):
                    ind = np.where(m)[0]
                    start, end = ind[0], ind[-1]
                    X[chain_i] = X[chain_i][start : end + 1]
                    S[chain_i] = S[chain_i][start : end + 1]
                    mask[chain_i] = mask[chain_i][start : end + 1]
                    mask_original[chain_i] = mask_original[chain_i][start : end + 1]
                    residue_idx[chain_i] = residue_idx[chain_i][start : end + 1]
                    chain_encoding_all[chain_i] = chain_encoding_all[chain_i][
                        start : end + 1
                    ]
                    for key in node_features.keys():
                        node_features[key][chain_i] = node_features[key][chain_i][
                            start : end + 1
                        ]
                    if len(cdr) > 0:
                        cdr[chain_i] = cdr[chain_i][start : end + 1]

            out = {}
            out["X"] = torch.from_numpy(np.concatenate(X, 0))
            out["S"] = torch.cat(S)
            out["mask"] = torch.from_numpy(np.concatenate(mask))
            out["mask_original"] = torch.from_numpy(np.concatenate(mask_original))
            out["chain_encoding_all"] = torch.cat(chain_encoding_all)
            out["residue_idx"] = torch.cat(residue_idx)
            out["chain_dict"] = chain_dict
            out["pdb_id"] = no_extension_name.split("-")[0]
            if len(cdr) != 0:
                out["cdr"] = torch.from_numpy(np.concatenate(cdr))
            for key, value_list in node_features.items():
                out[key] = torch.from_numpy(np.concatenate(value_list))
            with open(output_file, "wb") as f:
                pickle.dump(out, f)
        return output_names

    def set_cdr(self, cdr):
        """
        Set the CDR to be iterated over (only for SAbDab datasets).

        Parameters
        ----------
        cdr : {"H1", "H2", "H3", "L1", "L2", "L3"}
            The CDR to be iterated over. Set to `None` to go back to iterating over all chains.
        """

        if not self.sabdab:
            cdr = None
        if cdr == self.cdr:
            return
        self.cdr = cdr
        if cdr is None:
            self.indices = list(range(len(self.data)))
        else:
            self.indices = []
            print(f"Setting CDR to {cdr}...")
            for i, data in tqdm(enumerate(self.data)):
                if self.clusters is not None:
                    if data.split("__")[1] == cdr:
                        self.indices.append(i)
                else:
                    add = False
                    for chain in self.files[data]:
                        if chain.split("__")[1] == cdr:
                            add = True
                            break
                    if add:
                        self.indices.append(i)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        chain_id = None
        cdr = None
        idx = self.indices[idx]
        if self.clusters is None:
            id = self.data[idx]  # data is already filtered by length
            chain_id = random.choice(list(self.files[id].keys()))
            if self.cdr is not None:
                while self.cdr != chain_id.split("__")[1]:
                    chain_id = random.choice(list(self.files[id].keys()))
        else:
            cluster = self.data[idx]
            id = None
            chain_n = -1
            while (
                id is None or len(self.files[id][chain_id]) == 0
            ):  # some IDs can be filtered out by length
                if self.shuffle_clusters:
                    chain_n = random.randint(0, len(self.clusters[cluster]) - 1)
                else:
                    chain_n += 1
                id, chain_id = self.clusters[cluster][
                    chain_n
                ]  # get id and chain from cluster
        file = random.choice(self.files[id][chain_id])
        if "__" in chain_id:
            chain_id, cdr = chain_id.split("__")
        if self.loaded is None:
            with open(file, "rb") as f:
                try:
                    data = pickle.load(f)
                except EOFError:
                    print("EOFError", file)
                    raise
        else:
            data = deepcopy(self.loaded[file])
        data["chain_id"] = data["chain_dict"][chain_id]
        if cdr is not None:
            data["cdr_id"] = CDR[cdr]
        return data


def _download_dataset(tag, local_datasets_folder="./data/"):
    """
    Download the pre-processed data and the split dictionaries

    Parameters
    ----------
    tag : str
        name of the dataset (check `get_available_tags` to see the options)
    local_dataset_folder : str, default "./data/"
        the local folder that will contain proteinflow dataset folders, temporary files and logs

    Returns
    -------
    data_folder : str
        the path to the downloaded data folder
    """

    s3_data_path, s3_dict_path = _get_s3_paths_from_tag(tag)
    data_folder = os.path.join(local_datasets_folder, f"proteinflow_{tag}")
    dict_folder = os.path.join(
        local_datasets_folder, f"proteinflow_{tag}", "splits_dict"
    )

    print("Downloading dictionaries for splitting the dataset...")
    _download_dataset_dicts_from_s3(dict_folder, s3_dict_path)
    print("Done!")

    _download_dataset_from_s3(dataset_path=data_folder, s3_path=s3_data_path)
    return data_folder


def _biounits_in_clusters_dict(clusters_dict, excluded_files=None):
    """
    Return the list of all biounit files present in clusters_dict
    """

    if len(clusters_dict) == 0:
        return np.array([])
    if excluded_files is None:
        excluded_files = []
    return np.unique(
        [
            c[0]
            for c in list(np.concatenate(list(clusters_dict.values())))
            if c[0] not in excluded_files
        ]
    )


def _exclude(clusters_dict, set_to_exclude, exclude_based_on_cdr=None):
    """
    Exclude biounits from clusters_dict

    Parameters
    ----------
    clusters_dict : dict
        dictionary of clusters
    set_to_exclude : set
        set of biounits to exclude
    exclude_based_on_cdr : str, default None
        if not None, exclude based only on clusters based on this CDR (e.g. "H3")

    """
    excluded_set = set()
    excluded_dict = defaultdict(set)
    for cluster in list(clusters_dict.keys()):
        files = clusters_dict[cluster]
        exclude = False
        for biounit in files:
            if biounit in set_to_exclude:
                exclude = True
                break
        if exclude:
            if exclude_based_on_cdr is not None:
                if not cluster.endswith(exclude_based_on_cdr):
                    continue
            for biounit in files:
                clusters_dict[cluster].remove(biounit)
                excluded_dict[cluster].add(biounit)
                excluded_set.add(biounit)
    return excluded_dict, excluded_set


def _split_data(
    dataset_path="./data/proteinflow_20221110/",
    excluded_files=None,
    exclude_clusters=False,
    exclude_based_on_cdr=None,
):
    """
    Rearrange files into folders according to the dataset split dictionaries at `dataset_path/splits_dict`

    Parameters
    ----------
    dataset_path : str, default "./data/proteinflow_20221110/"
        The path to the dataset folder containing pre-processed entries and a `splits_dict` folder with split dictionaries (downloaded or generated with `get_split_dictionaries`)
    excluded_files : list, optional
        A list of files to exclude from the dataset
    exclude_clusters : bool, default False
        If True, exclude all files in a cluster if at least one file in the cluster is in `excluded_files`
    exclude_based_on_cdr : str, optional
        If not `None`, exclude all files in a cluster if the cluster name does not end with `exclude_based_on_cdr`
    """

    if excluded_files is None:
        excluded_files = []

    dict_folder = os.path.join(dataset_path, "splits_dict")
    with open(os.path.join(dict_folder, "train.pickle"), "rb") as f:
        train_clusters_dict = pickle.load(f)
    with open(os.path.join(dict_folder, "valid.pickle"), "rb") as f:
        valid_clusters_dict = pickle.load(f)
    with open(os.path.join(dict_folder, "test.pickle"), "rb") as f:
        test_clusters_dict = pickle.load(f)

    train_biounits = _biounits_in_clusters_dict(train_clusters_dict, excluded_files)
    valid_biounits = _biounits_in_clusters_dict(valid_clusters_dict, excluded_files)
    test_biounits = _biounits_in_clusters_dict(test_clusters_dict, excluded_files)
    train_path = os.path.join(dataset_path, "train")
    valid_path = os.path.join(dataset_path, "valid")
    test_path = os.path.join(dataset_path, "test")

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(valid_path):
        os.makedirs(valid_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    if len(excluded_files) > 0:
        set_to_exclude = set(excluded_files)
        excluded_files = set()
        excluded_clusters_dict = defaultdict(set)
        if exclude_clusters:
            for clusters_dict in [
                train_clusters_dict,
                valid_clusters_dict,
                test_clusters_dict,
            ]:
                subset_excluded_set, subset_excluded_dict = _exclude(
                    clusters_dict, set_to_exclude, exclude_based_on_cdr
                )
                excluded_files.update(subset_excluded_set)
                excluded_clusters_dict.update(subset_excluded_dict)
        excluded_files.update(set_to_exclude)
        excluded_clusters_dict = {k: list(v) for k, v in excluded_clusters_dict.items()}
        excluded_path = os.path.join(dataset_path, "excluded")
        if not os.path.exists(excluded_path):
            os.makedirs(excluded_path)
        print("Updating the split dictionaries...")
        with open(os.path.join(dict_folder, "train.pickle"), "wb") as f:
            pickle.dump(train_clusters_dict, f)
        with open(os.path.join(dict_folder, "excluded.pickle"), "wb") as f:
            pickle.dump(excluded_clusters_dict, f)
        print("Moving excluded files...")
        for biounit in tqdm(excluded_files):
            shutil.move(os.path.join(dataset_path, biounit), excluded_path)
    print("Moving files in the train set...")
    for biounit in tqdm(train_biounits):
        shutil.move(os.path.join(dataset_path, biounit), train_path)
    print("Moving files in the validation set...")
    for biounit in tqdm(valid_biounits):
        shutil.move(os.path.join(dataset_path, biounit), valid_path)
    print("Moving files in the test set...")
    for biounit in tqdm(test_biounits):
        shutil.move(os.path.join(dataset_path, biounit), test_path)


def _remove_database_redundancies(dir, seq_identity_threshold=0.9):
    """
    Remove all biounits in the database that are copies to another biounits in terms of sequence

    Sequence identity is defined by the 'seq_identity_threshold' parameter for robust detection of sequence similarity (missing residues, point mutations, ...).

    Parameters
    ----------
    dir : str
        the path to the database where all the biounits are stored in pickle files after their processing
    seq_identity_threshold : float, default .9
        the threshold that determines up to what percentage identity sequences are considered as the same

    Returns
    -------
    total_removed : int
        the total number of removed biounits
    """

    all_files = np.array(os.listdir(dir))
    all_pdbs = np.array([file[:4] for file in all_files])
    pdb_counts = Counter(all_pdbs)
    pdbs_to_check = [pdb for pdb in pdb_counts.keys() if pdb_counts[pdb] > 1]
    total_removed = []

    for pdb in tqdm(pdbs_to_check):
        biounits_list = np.array(
            [os.path.join(dir, file) for file in all_files[all_pdbs == pdb]]
        )
        biounits_list = sorted(biounits_list)
        redundancies = _check_biounits(biounits_list, seq_identity_threshold)
        if redundancies != []:
            for k in redundancies:
                total_removed.append(os.path.basename(biounits_list[k]).split(".")[0])
                subprocess.run(["rm", biounits_list[k]])

    return total_removed
