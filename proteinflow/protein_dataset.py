import os
import pickle
import random
import shutil
import subprocess
from collections import Counter, defaultdict
from copy import deepcopy
from itertools import combinations

import numpy as np
import torch
from p_tqdm import p_map
from torch.utils.data import Dataset
from tqdm import tqdm

from proteinflow.constants import ALPHABET, CDR_REVERSE, D3TO1, MAIN_ATOMS
from proteinflow.data import ProteinEntry
from proteinflow.pdb import _check_biounits
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
    should take a `proteinflow.data.ProteinEntry` instance and a list of chains and return a `numpy` array shaped as `(#residues, #features)`
    where `#residues` is the total number of residues in those chains and the features are concatenated in the order of the list:
    `func(data_entry: ProteinEntry, chains: list) -> np.ndarray`.

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
                    if len(classes_to_exclude) > 0:
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

    def _dihedral(self, data_entry, chains):
        """
        Dihedral angles
        """

        return data_entry.dihedral_angles(chains)

    def _sidechain(self, data_entry, chains):
        """
        Sidechain orientation (defined by the 'main atoms' in the `main_atom_dict` dictionary)
        """

        return data_entry.sidechain_orientation(chains)

    def _chemical(self, data_entry, chains):
        """
        Chemical features (hydropathy, volume, charge, polarity, acceptor/donor)
        """

        return data_entry.chemical_features(chains)

    def _sse(self, data_entry, chains):
        """
        Secondary structure features
        """

        return data_entry.secondary_structure(chains)

    def _sidechain_coords(self, data_entry, chains):
        """
        Sidechain coordinates
        """

        return data_entry.sidechain_coordinates(chains)

    def _process(self, filename, rewrite=False, max_length=None, min_cdr_length=None):
        """
        Process a proteinflow file and save it as ProteinMPNN features
        """

        input_file = os.path.join(self.dataset_folder, filename)
        no_extension_name = filename.split(".")[0]
        data_entry = ProteinEntry.from_pickle(input_file)
        chains = data_entry.get_chains()
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
        if self.cut_edges:
            data_entry.cut_missing_edges()
        for chains_i, chain_set in enumerate(chain_sets):
            output_file = os.path.join(
                self.features_folder, no_extension_name + f"_{chains_i}.pickle"
            )
            pass_set = False
            add_name = True
            if os.path.exists(output_file) and not rewrite:
                pass_set = True
                if max_length is not None:
                    if data_entry.get_length(chain_set) > max_length:
                        add_name = False
                if min_cdr_length is not None and data_entry.has_cdr():
                    cdr_length = data_entry.get_cdr_length(chain_set)
                    if not all(
                        [
                            length >= min_cdr_length
                            for length in cdr_length.values()
                            if length > 0
                        ]
                    ):
                        add_name = False
            else:
                if max_length is not None:
                    if data_entry.get_length(chains=chain_set) > max_length:
                        pass_set = True
                        add_name = False
                if min_cdr_length is not None and data_entry.has_cdr():
                    cdr_length = data_entry.get_cdr_length(chain_set)
                    if not all(
                        [
                            length >= min_cdr_length
                            for length in cdr_length.values()
                            if length > 0
                        ]
                    ):
                        add_name = False
                        pass_set = True

                if self.entry_type == "pair":
                    # intersect = []
                    if not data_entry.is_valid_pair(*chain_set):
                        # if not all(intersect):
                        pass_set = True
                        add_name = False
            if pass_set:
                continue

            out = {}
            out["pdb_id"] = no_extension_name.split("-")[0]
            out["mask_original"] = torch.tensor(data_entry.get_mask(chain_set))
            if self.interpolate != "none":
                data_entry.interpolate_coords(fill_ends=(self.interpolate == "all"))
            out["mask"] = torch.tensor(data_entry.get_mask(chain_set))
            out["S"] = torch.tensor(data_entry.get_sequence(chain_set, encode=True))
            out["X"] = torch.tensor(data_entry.get_coordinates(chain_set))
            out["residue_idx"] = torch.tensor(
                data_entry.get_index_array(chain_set, index_bump=100)
            )
            out["chain_encoding_all"] = torch.tensor(
                data_entry.get_chain_id_array(chain_set)
            )
            out["chain_dict"] = data_entry.get_chain_id_dict(chain_set)
            cdr_chain_set = set()
            if data_entry.has_cdr():
                out["cdr"] = data_entry.get_cdr(chain_set)
                chain_type_dict = data_entry.get_chain_type_dict(chain_set)
                if "heavy" in chain_type_dict:
                    cdr_chain_set.update(
                        [
                            f"{chain_type_dict['heavy']}__{cdr}"
                            for cdr in ["H1", "H2", "H3"]
                        ]
                    )
                if "light" in chain_type_dict:
                    cdr_chain_set.update(
                        [
                            f"{chain_type_dict['light']}__{cdr}"
                            for cdr in ["L1", "L2", "L3"]
                        ]
                    )
            for name in self.feature_types:
                if name not in self.feature_functions:
                    continue
                func = self.feature_functions[name]
                out[name] = func(data_entry, chain_set)

            if add_name:
                output_names.append(
                    (
                        os.path.basename(no_extension_name),
                        output_file,
                        chain_set if len(cdr_chain_set) == 0 else cdr_chain_set,
                    )
                )
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
            data["cdr_id"] = CDR_REVERSE[cdr]
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
