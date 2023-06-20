import os
import random as rd
import subprocess
from collections import defaultdict
from itertools import combinations

import networkx as nx
import numpy as np
from tqdm import tqdm

from proteinflow.sequences import (
    _create_pdb_seqs_dict,
    _load_pdbs,
    _merge_chains,
    _retrieve_seqs_names_list,
    _write_fasta,
)
from proteinflow.utils.common_utils import _find_correspondences, _test_availability


def _run_mmseqs2(fasta_file, tmp_folder, min_seq_id, cdr=None):
    """
    Run the MMSeqs2 command with the parameters we want

    Results are stored in the tmp_folder/MMSeqs2 directory.
    """

    folder = "MMSeqs2_results" if cdr is None else os.path.join("MMSeqs2_results", cdr)
    os.makedirs(os.path.join(tmp_folder, folder), exist_ok=True)
    method = "easy-linclust" if cdr is not None else "easy-cluster"
    args = [
        "mmseqs",
        method,
        fasta_file,
        os.path.join(tmp_folder, folder, "clusterRes"),
        os.path.join(tmp_folder, folder, "tmp"),
        "--min-seq-id",
        str(min_seq_id),
        "-v",
        "1",
    ]
    if cdr is not None:
        args += [
            "-k",
            "5",
            "--spaced-kmer-mode",
            "0",
            "--comp-bias-corr",
            "0",
            "--mask",
            "0",
        ]
    subprocess.run(args)


def _read_clusters(tmp_folder, cdr=None):
    """
    Read the output from MMSeqs2 and produces 2 dictionaries that store the clusters information

    In cluster_dict, values are the full names (pdb + chains) whereas in cluster_pdb_dict, values are just the PDB ids (so less clusters but bigger).
    """

    if cdr is None:
        cluster_file_fasta = os.path.join(
            tmp_folder, "MMSeqs2_results", "clusterRes_all_seqs.fasta"
        )
    else:
        cluster_file_fasta = os.path.join(
            tmp_folder, "MMSeqs2_results", cdr, "clusterRes_all_seqs.fasta"
        )
    with open(cluster_file_fasta) as f:
        cluster_dict = defaultdict(lambda: [])
        cluster_pdb_dict = defaultdict(lambda: [])
        cluster_name, sequence_name = None, None
        found_header = False

        for line in f.readlines():
            if line[0] == ">" and found_header:
                cluster_name = line[1:-1]
                sequence_name = line[1:-1]
                if cdr is not None:
                    cluster_name += "__" + cdr

            elif line[0] == ">":
                sequence_name = line[1:-1]
                found_header = True

            else:
                cluster_dict[cluster_name].append(sequence_name)
                cluster_pdb_dict[cluster_name].append(sequence_name[:4])
                found_header = False

        for k in cluster_pdb_dict.keys():
            cluster_pdb_dict[k] = np.unique(cluster_pdb_dict[k])

    return cluster_dict, cluster_pdb_dict


def _make_graph(cluster_pdb_dict):
    """
    Produces a graph that relates clusters together

    Connections represent a PDB shared by 2 clusters. The more shared PDBs, the stronger the connection.
    """

    keys = list(cluster_pdb_dict.keys())
    keys_mapping = {length: k for length, k in enumerate(keys)}
    adjacency_matrix = np.zeros((len(keys), len(keys)))

    seen_dict = defaultdict(set)
    for i, key in enumerate(keys):
        for pdb in cluster_pdb_dict[key]:
            seen_dict[pdb].add(i)
    for cluster_set in seen_dict.values():
        for i, j in combinations(cluster_set, 2):
            adjacency_matrix[i, j] += 1
            adjacency_matrix[j, i] += 1

    graph = nx.from_numpy_matrix(adjacency_matrix)
    nx.relabel_nodes(graph, keys_mapping, copy=False)
    return graph


def _check_for_heteromers(grouped_seqs, biounit_chains):
    """
    True if the chain names contained in grouped_seqs correspond to at least 2 different sequences
    """

    grouped_seqs = [group.split("-") for group in grouped_seqs]
    chain_seqs = [
        np.argmax([chain in group for group in grouped_seqs])
        for chain in biounit_chains
    ]
    return np.max(chain_seqs) != np.min(chain_seqs)


def _divide_according_to_chains_interactions(pdb_seqs_dict, dataset_dir):
    """
    Divide all the biounit chains into 3 groups: single chains, homomers and heteromers, depending on the other chains present or not in the biounit
    """

    heteromers = []
    homomers = []
    single_chains = []
    all_files = np.array(list(os.listdir(dataset_dir)))
    all_pdb_files = np.array([f[:4] for f in all_files])

    for pdb in tqdm(pdb_seqs_dict.keys()):
        file_names = all_files[all_pdb_files == pdb]
        if type(file_names) == str:
            file_names = [file_names]
        seqs = pdb_seqs_dict[pdb]
        if len(seqs) == 1 and len(seqs[0].split("-")) == 1:
            for file_name in file_names:
                single_chains.append((file_name, seqs[0]))

        elif len(seqs) == 1 and len(file_names) == 1:
            for chain in seqs[0].split("-"):
                homomers.append((file_names[0], chain))

        elif len(seqs) == 1:
            correspondences = _find_correspondences(file_names, dataset_dir)
            for biounit in correspondences.keys():
                if len(correspondences[biounit]) == 1:
                    single_chains.append((biounit, correspondences[biounit][0]))
                else:
                    for chain in correspondences[biounit]:
                        homomers.append((biounit, chain))

        else:
            correspondences = _find_correspondences(file_names, dataset_dir)
            for biounit in correspondences.keys():
                if len(correspondences[biounit]) == 1:
                    single_chains.append((biounit, correspondences[biounit][0]))
                elif _check_for_heteromers(seqs, correspondences[biounit]):
                    for chain in correspondences[biounit]:
                        heteromers.append((biounit, chain))
                else:
                    for chain in correspondences[biounit]:
                        homomers.append((biounit, chain))

    return single_chains, homomers, heteromers


def _find_chains_in_graph(
    graph, clusters_dict, biounit_chains_array, pdbs_array, chains_array
):
    """
    Find all the biounit chains present in a given graph or subgraph

    Return a dictionary for which each key is a cluster name (merged chains name) and the values are all the biounit chains contained in this cluster.
    """

    res_dict = {}
    for k, node in enumerate(graph):
        grouped_chains = clusters_dict[node]
        split_chains = np.concatenate(
            [
                [(pdb[:4], chain) for chain in pdb[5:].split("-")]
                for pdb in grouped_chains
            ]
        )
        biounits_chains = np.concatenate(
            [
                biounit_chains_array[
                    np.logical_and(pdbs_array == pdb, chains_array == chain)
                ]
                for pdb, chain in split_chains
            ]
        )

        res_dict[node] = biounits_chains

    return res_dict


def _find_repartition(chains_dict, homomers, heteromers):
    """
    Return a dictionary similar to the one created by find_chains_in_graph, with an additional level of classification for single chains, homomers and heteromers

    Dictionary structure : `{'single_chains' : {cluster_name : [biounit chains]}, 'homomers' : {cluster_name : [biounit chains]}, 'heteromers' : {cluster_name : [biounit chains]}}`.
    Additionally return the number of chains in each class (single chains, ...).
    """

    classes_dict = {
        "single_chains": defaultdict(lambda: []),
        "homomers": defaultdict(lambda: []),
        "heteromers": defaultdict(lambda: []),
    }
    n_single_chains, n_homomers, n_heteromers = 0, 0, 0

    for node in chains_dict.keys():
        for k, chain in enumerate(chains_dict[node]):
            if tuple(chain) in homomers:
                classes_dict["homomers"][node].append(chain)
                n_homomers += 1
            elif tuple(chain) in heteromers:
                classes_dict["heteromers"][node].append(chain)
                n_heteromers += 1
            else:
                classes_dict["single_chains"][node].append(chain)
                n_single_chains += 1

    for node in classes_dict["single_chains"]:
        classes_dict["single_chains"][node] = np.stack(
            classes_dict["single_chains"][node]
        )

    for node in classes_dict["homomers"]:
        classes_dict["homomers"][node] = np.stack(classes_dict["homomers"][node])

    for node in classes_dict["heteromers"]:
        classes_dict["heteromers"][node] = np.stack(classes_dict["heteromers"][node])

    return classes_dict, n_single_chains, n_homomers, n_heteromers


def _find_subgraphs_infos(
    subgraphs,
    clusters_dict,
    biounit_chains_array,
    pdbs_array,
    chains_array,
    homomers,
    heteromers,
):
    """
    Given a list of subgraphs, return a list of dictionaries and an array of sizes of the same length

    Dictionaries are the `chains_dict` and `classes_dict` corresponding to each subgraph, returned by the `find_chains_in_graph`
    and `find_repartition` functions respectively. The array of sizes is of shape (len(subgraph), 3). It gives the number of single chains,
    homomers and heteromers present in each subgraph.
    """

    size_array = np.zeros((len(subgraphs), 3))
    dict_list = []
    for k, subgraph in tqdm(enumerate(subgraphs)):
        chains_dict = _find_chains_in_graph(
            subgraph, clusters_dict, biounit_chains_array, pdbs_array, chains_array
        )
        classes_dict, n_single_chains, n_homomers, n_heteromers = _find_repartition(
            chains_dict, homomers, heteromers
        )
        size_array[k] = np.array([n_single_chains, n_homomers, n_heteromers])
        dict_list.append((chains_dict, classes_dict))

    return (
        dict_list,
        size_array,
        int(np.sum(size_array[:, 0])),
        int(np.sum(size_array[:, 1])),
        int(np.sum(size_array[:, 2])),
    )


def _construct_dataset(dict_list, size_array, indices):
    """
    Get a supergraph containing all subgraphs indicated by `indices`

    Given the `dict_list` and `size_array` returned by `find_subgraphs_info`, return the 2 dictionaries (`chains_dict` and `classes_dict`)
    corresponding to the graph encompassing all the subgraphs indicated by indices.
    Additionally return the number of single chains, homomers and heteromers in this supergraph.
    """

    dataset_clusters_dict = {}
    dataset_classes_dict = {"single_chains": {}, "homomers": {}, "heteromers": {}}
    single_chains_size, homomers_size, heteromers_size = 0, 0, 0
    for k in indices:
        chains_dict, classes_dict = dict_list[k]
        n_single_chains, n_homomers, n_heteromers = size_array[k]
        single_chains_size += n_single_chains
        homomers_size += n_homomers
        heteromers_size += n_heteromers

        dataset_clusters_dict.update(chains_dict)
        for chain_class in classes_dict.keys():
            dataset_classes_dict[chain_class].update(classes_dict[chain_class])

    return (
        dataset_clusters_dict,
        dataset_classes_dict,
        single_chains_size,
        homomers_size,
        heteromers_size,
    )


def _remove_elements_from_dataset(
    indices,
    remaining_indices,
    chain_class,
    size_obj,
    current_sizes,
    size_array,
    tolerance=0.2,
):
    """
    Remove values from indices until we get the required (`size_obj`) number of chains in the class of interest (`chain_class`)

    Parameter `chain_class` corresponds to the single chain (0), homomer (1) or heteromer (2) class.
    """

    sizes = [s[chain_class] for s in size_array[indices]]
    sorted_sizes_indices = np.argsort(sizes)[::-1]

    while current_sizes[chain_class] > size_obj and len(sorted_sizes_indices) > 0:
        if (
            current_sizes[chain_class]
            - size_array[indices[sorted_sizes_indices[0]], chain_class]
            < (1 - tolerance) * size_obj
        ):
            sorted_sizes_indices = sorted_sizes_indices[1:]
            continue

        current_sizes -= size_array[indices[sorted_sizes_indices[0]]]
        remaining_indices.append(indices[sorted_sizes_indices[0]])
        indices.pop(sorted_sizes_indices[0])
        sizes = [s[chain_class] for s in size_array[indices]]
        sorted_sizes_indices = np.argsort(sizes)[::-1]

    return (
        indices,
        remaining_indices,
        current_sizes[0],
        current_sizes[1],
        current_sizes[2],
    )


def _check_mmseqs():
    """
    Raise an error if MMseqs2 is not installed
    """

    devnull = open(os.devnull, "w")
    retval = subprocess.call(
        ["mmseqs", "--help"], stdout=devnull, stderr=subprocess.STDOUT
    )
    devnull.close()
    if retval != 0:
        raise RuntimeError(
            "Please install the MMseqs2 library following the instructions at https://github.com/soedinglab/MMseqs2 (recommended: conda)"
        )


def _add_elements_to_dataset(
    indices,
    remaining_indices,
    chain_class,
    size_obj,
    current_sizes,
    size_array,
    tolerance=0.2,
):
    """
    Add values to indices until we get the required (`size_obj`) number of chains in the class of interest (`chain_class`)

    Parameter `chain_class` corresponds to the single chain (0), homomer (1) or heteromer (2) class.
    """

    sizes = [s[chain_class] for s in size_array[remaining_indices]]
    sorted_sizes_indices = np.argsort(sizes)[::-1]

    while current_sizes[chain_class] < size_obj and len(sorted_sizes_indices) > 0:
        if (
            current_sizes[chain_class]
            + size_array[remaining_indices[sorted_sizes_indices[0]], chain_class]
            > (1 + tolerance) * size_obj
        ):
            sorted_sizes_indices = sorted_sizes_indices[1:]
            continue

        current_sizes += size_array[remaining_indices[sorted_sizes_indices[0]]].astype(
            int
        )
        indices.append(remaining_indices[sorted_sizes_indices[0]])
        remaining_indices.pop(sorted_sizes_indices[0])
        sizes = [s[chain_class] for s in size_array[remaining_indices]]
        sorted_sizes_indices = np.argsort(sizes)[::-1]

    return (
        indices,
        remaining_indices,
        current_sizes[0],
        current_sizes[1],
        current_sizes[2],
    )


def _adjust_dataset(
    indices,
    remaining_indices,
    dict_list,
    size_array,
    n_single_chains,
    n_homomers,
    n_heteromers,
    single_chains_size,
    homomers_size,
    heteromers_size,
    sc_available,
    hm_available,
    ht_available,
    tolerance=0.2,
):
    """
    If required, remove and add values in indices so that the number of chains in each class correspond to the required numbers within a tolerance

    First remove and then add (if necessary, for each class separately).
    In the end, we might end up with more chains than desired in the first 2 classes but for a reasonable tolerance (~10-20 %), this should not happen.
    """

    if single_chains_size > (1 + tolerance) * n_single_chains and sc_available:
        (
            indices,
            remaining_indices,
            single_chains_size,
            homomers_size,
            heteromers_size,
        ) = _remove_elements_from_dataset(
            indices,
            remaining_indices,
            0,
            n_single_chains,
            np.array([single_chains_size, homomers_size, heteromers_size]),
            size_array,
            tolerance=tolerance,
        )

    if homomers_size > (1 + tolerance) * n_homomers and hm_available:
        (
            indices,
            remaining_indices,
            single_chains_size,
            homomers_size,
            heteromers_size,
        ) = _remove_elements_from_dataset(
            indices,
            remaining_indices,
            1,
            n_homomers,
            np.array([single_chains_size, homomers_size, heteromers_size]),
            size_array,
            tolerance=tolerance,
        )

    if heteromers_size > (1 + tolerance) * n_heteromers and ht_available:
        (
            indices,
            remaining_indices,
            single_chains_size,
            homomers_size,
            heteromers_size,
        ) = _remove_elements_from_dataset(
            indices,
            remaining_indices,
            2,
            n_heteromers,
            np.array([single_chains_size, homomers_size, heteromers_size]),
            size_array,
            tolerance=tolerance,
        )

    if single_chains_size < (1 - tolerance) * n_single_chains and sc_available:
        (
            indices,
            remaining_indices,
            single_chains_size,
            homomers_size,
            heteromers_size,
        ) = _add_elements_to_dataset(
            indices,
            remaining_indices,
            0,
            n_single_chains,
            np.array([single_chains_size, homomers_size, heteromers_size]),
            size_array,
            tolerance=tolerance,
        )

    if homomers_size < (1 - tolerance) * n_homomers and hm_available:
        (
            indices,
            remaining_indices,
            single_chains_size,
            homomers_size,
            heteromers_size,
        ) = _add_elements_to_dataset(
            indices,
            remaining_indices,
            1,
            n_homomers,
            np.array([single_chains_size, homomers_size, heteromers_size]),
            size_array,
            tolerance=tolerance,
        )

    if heteromers_size < (1 - tolerance) * n_heteromers and ht_available:
        (
            indices,
            remaining_indices,
            single_chains_size,
            homomers_size,
            heteromers_size,
        ) = _add_elements_to_dataset(
            indices,
            remaining_indices,
            2,
            n_heteromers,
            np.array([single_chains_size, homomers_size, heteromers_size]),
            size_array,
            tolerance=tolerance,
        )

    (
        dataset_clusters_dict,
        dataset_classes_dict,
        single_chains_size,
        homomers_size,
        heteromers_size,
    ) = _construct_dataset(dict_list, size_array, indices)
    return (
        dataset_clusters_dict,
        dataset_classes_dict,
        single_chains_size,
        homomers_size,
        heteromers_size,
        remaining_indices,
    )


def _fill_dataset(
    dict_list,
    size_array,
    n_samples,
    n_single_chains,
    n_homomers,
    n_heteromers,
    remaining_indices,
    n_max_iter=50,
    tolerance=0.2,
):
    """
    Construct a dataset from subgraphs indicated by `indices`

    Given a list of indices to choose from (`remaining_indices`), choose a list of subgraphs to construct a dataset containing the required number of
    biounits for each class (single chains, ...) within a tolerance.
    Return the same outputs as the construct_dataset function, as long as the list of remaining indices after selection.
    """

    single_chains_size, homomers_size, heteromers_size = 0, 0, 0
    sc_available, hm_available, ht_available = _test_availability(
        size_array, n_samples
    )  # rule of thumb to estimate if it is logical to try to fill the dataset with a given class
    distribution_satisfied = False
    n_iter = 0

    while not distribution_satisfied and n_iter < n_max_iter:
        n_iter += 1
        indices = rd.sample(remaining_indices, n_samples)
        (
            dataset_clusters_dict,
            dataset_classes_dict,
            single_chains_size,
            homomers_size,
            heteromers_size,
        ) = _construct_dataset(dict_list, size_array, indices)
        distribution_satisfied = (
            (single_chains_size > (1 - tolerance) * n_single_chains or not sc_available)
            and (
                single_chains_size < (1 + tolerance) * n_single_chains
                or not sc_available
            )
            and (homomers_size > (1 - tolerance) * n_homomers or not hm_available)
            and (homomers_size < (1 + tolerance) * n_homomers or not hm_available)
            and (heteromers_size > (1 - tolerance) * n_heteromers or not ht_available)
            and (heteromers_size < (1 + tolerance) * n_heteromers or not ht_available)
        )

    if not distribution_satisfied:
        (
            dataset_clusters_dict,
            dataset_classes_dict,
            single_chains_size,
            homomers_size,
            heteromers_size,
            remaining_indices,
        ) = _adjust_dataset(
            indices,
            [i for i in remaining_indices if i not in indices],
            dict_list,
            size_array,
            n_single_chains,
            n_homomers,
            n_heteromers,
            single_chains_size,
            homomers_size,
            heteromers_size,
            sc_available,
            hm_available,
            ht_available,
            tolerance=tolerance,
        )
    else:
        remaining_indices = [i for i in remaining_indices if i not in indices]

    print("Number of samplings (fill_dataset):", n_iter)
    return (
        dataset_clusters_dict,
        dataset_classes_dict,
        remaining_indices,
        single_chains_size,
        homomers_size,
        heteromers_size,
    )


def _get_subgraph_files(
    subgraphs,
    clusters_dict,
    pdb_arr,
    chain_arr,
    files_arr,
):
    """
    Given a list of subgraphs, return a dictionary of the form {cluster: [(filename, chain__cdr)]}
    """

    out = {}  # cluster: [(file, chain__cdr)]
    for subgraph in subgraphs:
        for cluster in subgraph.nodes:
            chains = []
            _, cdr = cluster.split("__")
            for chain in clusters_dict[cluster]:
                pdb, chain_ids = chain.split("_")
                for chain_id in chain_ids.split("-"):
                    mask = (pdb_arr == pdb) & (chain_arr == chain_id)
                    chains += [(x, chain_id + "__" + cdr) for x in files_arr[mask]]
            out[cluster] = chains
    return out


def _split_subgraphs(
    lengths,
    num_clusters_valid,
    num_clusters_test,
    tolerance,
):
    """
    Split the list of subgraphs into three sets (train, valid, test) according to the number of biounits in each subgraph
    """

    for _ in range(50):
        indices = np.random.permutation(np.arange(1, len(lengths)))
        valid_indices = []
        test_indices = []
        train_indices = [0]
        valid_sum = 0
        test_sum = 0
        for i in indices:
            if valid_sum < num_clusters_valid:
                if (
                    valid_sum < num_clusters_valid * (1 - tolerance)
                    or lengths[i] < tolerance * num_clusters_valid
                ):
                    valid_indices.append(i)
                    valid_sum += lengths[i]
                    continue
            if test_sum < num_clusters_test:
                if (
                    test_sum < num_clusters_test * (1 - tolerance)
                    or lengths[i] < tolerance * num_clusters_test
                ):
                    test_indices.append(i)
                    test_sum += lengths[i]
                    continue
            train_indices.append(i)
        valid_ok = valid_sum >= num_clusters_valid * (
            1 - tolerance
        ) and valid_sum <= num_clusters_valid * (1 + tolerance)
        test_ok = test_sum >= num_clusters_test * (
            1 - tolerance
        ) and test_sum <= num_clusters_test * (1 + tolerance)
        if valid_ok and test_ok:
            break
    return train_indices, valid_indices, test_indices


def _split_dataset_with_graphs(
    graph,
    clusters_dict,
    merged_seqs_dict,
    dataset_dir,
    valid_split=0.05,
    test_split=0.05,
    tolerance=0.2,
):
    """
    Given a graph representing connections between MMSeqs2 clusters, split the dataset between train, validation and test sets.
    Each onnected component of the graph is considered as a group.
    Then, groups are split into the 3 sets so that each set has the right amount of biounits.
    It has been observed that the biggest group represents about 15-20 % of all the biounits and thus it is automatically assigned to the train set.
    It is difficult to have the exact ratio of biounits in each set since biounits are manipulated by groups.
    However, within an acceptable tolerance (default 20 % of the split ratio - but in theory it can be smaller), the split ratios are respected.
    The process first try to randomly assign the groups to a set.
    If after 50 trials the partition fails to comply to the requirements, the last partition is kept and small adjustments are made by moving groups from sets one by one until the ratios are approximately respected.
    Note that for very small datasets (around 250 biounits), this method will probably fail. But it also does not make much sense to use it for so few data.

    Parameters
    ----------
    graph : networkx graph
        the graph representing connections between MMSeqs2 clusters
    clusters_dict : dict
        the dictionary containing all the biounit files organized by clusters
    merged_seqs_dict : dict
        the dictionary containing all the merged (by similarity and PDB id) chains organized by cluster
    dataset_dir : str
        the path to the dataset
    valid_split : float, default 0.05
        the validation split ratio
    test_split : float, default 0.05
        the test split ratio

    Returns
    -------
    train_clusters_dict : dict
        the dictionary containing all the clusters (keys) and the biounit chains they contain for the training dataset
        structure : `{cluster_id : [(biounit_file_name, chain), (..., ...), ...]}`
    train_classes_dict : dict
        the same dictionary as train_cluster_dict with an additional level of classification (by single chains, homomers, heteromers)
        structure : `{'single_chains' : train_cluster_dict_like_dict, 'homomers' : train_cluster_dict_like_dict, 'heteromers' : train_cluster_dict_like_dict}`
    valid_clusters_dict : dict
        see train_clusters_dict but for validation set
    valid_classes_dict : dict
        see train_classes_dict but for validation set
    test_clusters_dict : dict
        see train_clusters_dict but for test set
    test_classes_dict : dict
        see train_classes_dict but for test set
    single_chains : list
        the list of all biounit chains (string names) that are in a single chain state (in their biounit)
    homomers : list
        the list of all biounit chains (string names) that are in a homomeric state (in their biounit)
    heteromers : list
        the list of all biounit chains (string names) that are in a heteromeric state (in their biounit)
    """

    sample_cluster = list(clusters_dict.keys())[0]
    sabdab = "__" in sample_cluster

    subgraphs = np.array(
        [
            graph.subgraph(c)
            for c in sorted(nx.connected_components(graph), key=len, reverse=True)
        ],
        dtype=object,
    )

    if not sabdab:
        remaining_indices = list(np.arange(1, len(subgraphs)))
        seqs_names_list = _retrieve_seqs_names_list(merged_seqs_dict)
        pdb_seqs_dict = _create_pdb_seqs_dict(seqs_names_list)
        single_chains, homomers, heteromers = _divide_according_to_chains_interactions(
            pdb_seqs_dict, dataset_dir
        )
        biounit_chains_array = np.array(single_chains + homomers + heteromers)
        pdbs_array = np.array([c[0][:4] for c in biounit_chains_array])
        chains_array = np.array([c[1] for c in biounit_chains_array])

        (
            dict_list,
            size_array,
            n_single_chains,
            n_homomers,
            n_heteromers,
        ) = _find_subgraphs_infos(
            subgraphs,
            clusters_dict,
            biounit_chains_array,
            pdbs_array,
            chains_array,
            homomers,
            heteromers,
        )

        (
            n_single_chains_valid,
            n_homomers_valid,
            n_heteromers_valid,
        ) = valid_split * np.array([n_single_chains, n_homomers, n_heteromers])
        (
            n_single_chains_test,
            n_homomers_test,
            n_heteromers_test,
        ) = test_split * np.array([n_single_chains, n_homomers, n_heteromers])
        n_samples_valid, n_samples_test = int(valid_split * len(subgraphs)), int(
            test_split * len(subgraphs)
        )

        (
            valid_clusters_dict,
            valid_classes_dict,
            remaining_indices,
            n_single_chains_valid,
            n_homomers_valid,
            n_heteromers_valid,
        ) = _fill_dataset(
            dict_list,
            size_array,
            n_samples_valid,
            n_single_chains_valid,
            n_homomers_valid,
            n_heteromers_valid,
            remaining_indices,
            tolerance=tolerance,
        )

        (
            test_clusters_dict,
            test_classes_dict,
            remaining_indices,
            n_single_chains_test,
            n_homomers_test,
            n_heteromers_test,
        ) = _fill_dataset(
            dict_list,
            size_array,
            n_samples_test,
            n_single_chains_test,
            n_homomers_test,
            n_heteromers_test,
            remaining_indices,
            tolerance=tolerance,
        )

        remaining_indices.append(
            0
        )  # add the big first cluster, that we always want in the training set
        (
            train_clusters_dict,
            train_classes_dict,
            n_single_chains_train,
            n_homomers_train,
            n_heteromers_train,
        ) = _construct_dataset(dict_list, size_array, remaining_indices)

        print("Classes distribution (single chain / homomer / heteromer):")
        print(
            "Train set:",
            int(n_single_chains_train),
            "/",
            int(n_homomers_train),
            "/",
            int(n_heteromers_train),
        )
        print(
            "Validation set:",
            int(n_single_chains_valid),
            "/",
            int(n_homomers_valid),
            "/",
            int(n_heteromers_valid),
        )
        print(
            "Test set:",
            int(n_single_chains_test),
            "/",
            int(n_homomers_test),
            "/",
            int(n_heteromers_test),
        )

    else:
        n_samples_valid, n_samples_test = int(valid_split * len(clusters_dict)), int(
            test_split * len(clusters_dict)
        )
        train_indices, val_indices, test_indices = _split_subgraphs(
            [len(x) for x in subgraphs], n_samples_valid, n_samples_test, tolerance
        )
        files_arr = []
        pdb_arr = []
        chain_arr = []
        for file in os.listdir(dataset_dir):
            if not file.endswith(".pickle"):
                continue
            chains = [
                x for x in file.split("-")[1].split(".")[0].split("_") if x != "nan"
            ]
            chain_arr += chains
            pdb_arr += [file.split("-")[0]] * len(chains)
            files_arr += [file] * len(chains)
        files_arr = np.array(files_arr)
        pdb_arr = np.array(pdb_arr)
        chain_arr = np.array(chain_arr)
        train_clusters_dict = _get_subgraph_files(
            subgraphs=subgraphs[train_indices],
            clusters_dict=clusters_dict,
            files_arr=files_arr,
            pdb_arr=pdb_arr,
            chain_arr=chain_arr,
        )
        valid_clusters_dict = _get_subgraph_files(
            subgraphs=subgraphs[val_indices],
            clusters_dict=clusters_dict,
            files_arr=files_arr,
            pdb_arr=pdb_arr,
            chain_arr=chain_arr,
        )
        test_clusters_dict = _get_subgraph_files(
            subgraphs=subgraphs[test_indices],
            clusters_dict=clusters_dict,
            files_arr=files_arr,
            pdb_arr=pdb_arr,
            chain_arr=chain_arr,
        )
        train_classes_dict, valid_classes_dict, test_classes_dict = {}, {}, {}
        single_chains, homomers, heteromers = [], [], []

    return (
        train_clusters_dict,
        train_classes_dict,
        valid_clusters_dict,
        valid_classes_dict,
        test_clusters_dict,
        test_classes_dict,
        single_chains,
        homomers,
        heteromers,
    )


def _build_dataset_partition(
    dataset_dir,
    tmp_folder,
    valid_split=0.05,
    test_split=0.05,
    tolerance=0.2,
    min_seq_id=0.3,
    sabdab=False,
):
    """Build training, validation and test sets from a curated dataset of biounit, using MMSeqs2 for clustering.

    Parameters
    ----------
    dataset_dir : str
        the path to the dataset
    valid_split : float in [0, 1], default 0.05
        the validation split ratio
    test_split : float in [0, 1], default 0.05
        the test split ratio
    min_seq_id : float in [0, 1], default 0.3
        minimum sequence identity for `mmseqs`
    sabdab : bool, default False
        whether the dataset is the SAbDab dataset or not

    Output
    ------
    train_clusters_dict : dict
        the dictionary containing all the clusters (keys) and the biounit chains they contain for the training dataset
        structure : {cluster_id : [(biounit_file_name, chain), (..., ...), ...]}
    train_classes_dict : dict
        the same dictionary as train_cluster_dict with an additional level of classification (by single chains, homomers, heteromers)
        structure : {'single_chains' : train_cluster_dict_like_dict, 'homomers' : train_cluster_dict_like_dict, 'heteromers' : train_cluster_dict_like_dict}
    valid_clusters_dict : dict
        see train_clusters_dict but for validation set
    valid_classes_dict : dict
        see train_classes_dict but for validation set
    test_clusters_dict : dict
        see train_clusters_dict but for test set
    test_classes_dict : dict
        see train_classes_dict but for test set

    """
    cdrs = ["L1", "L2", "L3", "H1", "H2", "H3"] if sabdab else [None]
    for cdr in cdrs:
        if cdr is not None:
            print(f"Clustering with MMSeqs2 for CDR {cdr}...")
        else:
            print("Clustering with MMSeqs2...")
        # retrieve all sequences and create a merged_seqs_dict
        merged_seqs_dict = _load_pdbs(
            dataset_dir, cdr=cdr
        )  # keys: pdb_id, values: list of chains and sequences
        lengths = []
        for k, v in merged_seqs_dict.items():
            lengths += [len(x[1]) for x in v]
        merged_seqs_dict = _merge_chains(merged_seqs_dict)  # remove redundant chains

        # write sequences to a fasta file for clustering with MMSeqs2, run MMSeqs2 and delete the fasta file
        fasta_file = os.path.join(tmp_folder, "all_seqs.fasta")
        _write_fasta(
            fasta_file, merged_seqs_dict
        )  # write all sequences from merged_seqs_dict to fasta file
        _run_mmseqs2(
            fasta_file, tmp_folder, min_seq_id, cdr=cdr
        )  # run MMSeqs2 on fasta file
        subprocess.run(["rm", fasta_file])

    # retrieve MMSeqs2 clusters and build a graph with these clusters
    clusters_dict = {}
    clusters_pdb_dict = {}
    for cdr in cdrs:
        c_dict, c_pdb_dict = _read_clusters(
            tmp_folder=tmp_folder,
            cdr=cdr,
        )
        clusters_dict.update(c_dict)
        clusters_pdb_dict.update(c_pdb_dict)

    subprocess.run(["rm", "-r", os.path.join(tmp_folder, "MMSeqs2_results")])
    graph = _make_graph(clusters_pdb_dict)

    # import pickle

    # with open("graph.pickle", "wb") as f:
    #     pickle.dump(graph, f)

    # perform the splitting into train, validation and tesst sets
    (
        train_clusters_dict,
        train_classes_dict,
        valid_clusters_dict,
        valid_classes_dict,
        test_clusters_dict,
        test_classes_dict,
        *_,
    ) = _split_dataset_with_graphs(
        graph,
        clusters_dict,
        merged_seqs_dict,
        dataset_dir,
        valid_split=valid_split,
        test_split=test_split,
        tolerance=tolerance,
    )

    return (
        train_clusters_dict,
        train_classes_dict,
        valid_clusters_dict,
        valid_classes_dict,
        test_clusters_dict,
        test_classes_dict,
    )
