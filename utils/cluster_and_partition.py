import os
import copy
import subprocess
import editdistance
import numpy as np
import random as rd
import pickle as pkl
import networkx as nx
from tqdm import tqdm
from collections import defaultdict



def unique_chains(seqs_list):

    new_seqs_list = [seqs_list[0]]
    chains = [new_seqs_list[0][0]]

    for seq in seqs_list[1 : ]:
        if seq[0] not in chains:
            new_seqs_list.append(seq)
            chains.append(seq[0])
    
    return new_seqs_list


def merge_chains(seqs_dict_):

    """
    Look into the chains of each PDB and regroup redundancies (at 90% sequence identity)
    """

    seqs_dict = copy.deepcopy(seqs_dict_)
    pdbs_to_delete = []

    for pdb in tqdm(seqs_dict.keys()):

        if seqs_dict[pdb] == []:
            pdbs_to_delete.append(pdb)
            continue

        seqs_dict[pdb] = unique_chains(seqs_dict[pdb])
        groups, ref_seqs, indexes = [], [], []

        for k in range(len(seqs_dict[pdb])):
            
            if k in indexes:
                continue
            group = [seqs_dict[pdb][k][0]]
            ref_seq = seqs_dict[pdb][k][1]
            ref_seqs.append(ref_seq)
            indexes.append(k)

            for l in range(k + 1, len(seqs_dict[pdb])):
                chain, seq = seqs_dict[pdb][l][0], seqs_dict[pdb][l][1]
                if l in indexes or len(seq) > 1.1 * len(ref_seq) or len(seq) < .9 * len(ref_seq) or editdistance.eval(seq, ref_seq) / max(len(seq), len(ref_seq)) > .1:
                    continue
                group.append(chain)
                indexes.append(l)
            
            groups.append(group)

        new_group = []
        for group, seq in zip(groups, ref_seqs):
            new_group.append(('-'.join(group), seq))
        seqs_dict[pdb] = new_group
    
    for pdb in pdbs_to_delete:
        del seqs_dict[pdb]
    
    return seqs_dict


def load_pdbs(dir):

    """
    Load biounits and group their sequences by PDB and similarity (90%)
    """

    seqs_dict = defaultdict(lambda : [])

    for file in tqdm(os.listdir(dir)):

        load_path = os.path.join(dir, file)
        with open(load_path, 'rb') as f:
            pdb_dict = pkl.load(f)
        seqs = [(chain, pdb_dict[chain]['seq']) for chain in pdb_dict.keys()]
        seqs_dict[file[ : 4]] += seqs
    
    return seqs_dict


def write_fasta(fasta_path, merged_seqs_dict):

    """
    Write a fasta file containing all the sequences contained in the merged_seqs_dict dictionary
    """

    with open(fasta_path, 'w') as f:

        for k in merged_seqs_dict.keys():
            for chain, seq in merged_seqs_dict[k]:
                f.write('>' + k + '_' + chain + '\n')
                f.write(seq + '\n')


def run_mmseqs2(fasta_file, tmp_folder):

    """
    Run the MMSeqs2 command with the parameters we want
    Results are stored in the tmp_folder/MMSeqs2 directory
    """

    os.makedirs(os.path.join(tmp_folder, 'MMSeqs2_results'))
    subprocess.run(['mmseqs', 'easy-cluster', fasta_file, os.path.join(tmp_folder, 'MMSeqs2_results/clusterRes'), os.path.join(tmp_folder, 'MMSeqs2_results/tmp'), '--min-seq-id', '0.3'])


def read_clusters(cluster_file_fasta):

    """
    Read the output from MMSeqs2 and produces 2 dictionaries that store the clusters information.
    In cluster_dict, keys are the full names (pdb + chains) whereas in cluster_pdb_dict, keys are just the PDB ids (so less clusters but bigger)
    """

    with open(cluster_file_fasta, 'r') as f:

        cluster_dict = defaultdict(lambda : [])
        cluster_pdb_dict = defaultdict(lambda : [])
        cluster_name, sequence_name = None, None
        found_header = False

        for line in f.readlines():
            if line[0] == '>' and found_header:
                cluster_name = line[1 : -2]
                sequence_name = line[1 : -2]
            
            elif line[0] == '>':
                sequence_name = line[1 : -2]
                found_header = True
            
            else:
                cluster_dict[cluster_name].append(sequence_name)
                cluster_pdb_dict[cluster_name].append(sequence_name[:4])
                found_header = False
        
        for k in cluster_pdb_dict.keys():
            cluster_pdb_dict[k] = np.unique(cluster_pdb_dict[k])
    
    return cluster_dict, cluster_pdb_dict


def make_graph(cluster_pdb_dict):

    """
    Produces a graph that relates clusters together.
    Connections represent a PDB shared by 2 clusters. The more shared PDBs, the stronger the connection.
    """

    keys = list(cluster_pdb_dict.keys())
    keys_dict = {k : l for l, k in enumerate(keys)}
    keys_mapping = {l : k for l, k in enumerate(keys)}
    adjacency_matrix = np.zeros((len(keys), len(keys)))

    for k, key1 in enumerate(keys):
        for pdb in cluster_pdb_dict[key1]:
            for key2 in keys[k + 1 : ]:
                if pdb in cluster_pdb_dict[key2]:
                    adjacency_matrix[keys_dict[key1], keys_dict[key2]] += 1
                    adjacency_matrix[keys_dict[key2], keys_dict[key1]] += 1
    
    graph = nx.from_numpy_matrix(adjacency_matrix)
    nx.relabel_nodes(graph, keys_mapping, copy=False)
    return graph


def retrieve_seqs_names_list(merged_seqs_dict):

    """
    Retrieve all the sequences names that are contained in the merged_seqs_dict (same names as in the fasta file in input to MMSeqs2)
    """

    seqs_names = []
    for k in merged_seqs_dict.keys():
        for chain, _ in merged_seqs_dict[k]:
            seqs_names.append(k + '_' + chain)
    
    return seqs_names


def create_pdb_seqs_dict(seqs_names_list):

    """
    Return a dictionary that has the PDB ids as keys and the list of all the chain names that correspond to this PDB
    """

    pdb_seqs_dict = defaultdict(lambda : [])
    for name in seqs_names_list:
        pdb_seqs_dict[name[ : 4]].append(name[5 : ])
    
    return pdb_seqs_dict


def find_correspondances(files, dataset_dir):

    """
    Return a dictionary that contains all the biounits in the database (keys) and the list of all the chains that are in these biounits (values)
    """

    correspondances = defaultdict(lambda : [])
    for file in files:
        biounit = file
        with open(os.path.join(dataset_dir, file), 'rb') as f:
            keys = pkl.load(f)
            for k in keys:
                correspondances[biounit].append(k)
    
    return correspondances


def check_for_heteromers(grouped_seqs, biounit_chains):

    """
    True if the chain names contained in grouped_seqs correspond to at least 2 different sequences
    """

    grouped_seqs = [group.split('-') for group in grouped_seqs]
    chain_seqs = [np.argmax([chain in group for group in grouped_seqs]) for chain in biounit_chains]
    return np.max(chain_seqs) != np.min(chain_seqs)


def divide_according_to_chains_interactions(pdb_seqs_dict, dataset_dir):

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
        if len(seqs) == 1 and len(seqs[0].split('-')) == 1:
            single_chains.append((file_names[0], seqs[0]))
        
        elif len(seqs) == 1 and len(file_names) == 1:
            for chain in seqs[0].split('-'):
                homomers.append((file_names[0], chain))
        
        elif len(seqs) == 1:

            correspondances = find_correspondances(file_names, dataset_dir)
            for biounit in correspondances.keys():
                if len(correspondances[biounit]) == 1:
                    single_chains.append((biounit, correspondances[biounit][0]))
                else:
                    for chain in correspondances[biounit]:
                        homomers.append((biounit, chain))
        
        else:

            correspondances = find_correspondances(file_names, dataset_dir)
            for biounit in correspondances.keys():
                if len(correspondances[biounit]) == 1:
                    single_chains.append((biounit, correspondances[biounit][0]))
                elif check_for_heteromers(seqs, correspondances[biounit]):
                    for chain in correspondances[biounit]:
                        heteromers.append((biounit, chain))
                else:
                    for chain in correspondances[biounit]:
                        homomers.append((biounit, chain))
    
    return single_chains, homomers, heteromers


def find_chains_in_graph(graph, clusters_dict, biounit_chains_array, pdbs_array, chains_array):

    """
    Find all the biounit chains present in a given graph or subgraph.
    Return a dictionary for which each key is a cluster name (merged chains name) and the values are all the biounit chains contained in this cluster
    """

    res_dict = {}
    for node in graph:
        grouped_chains = clusters_dict[node]
        split_chains = np.concatenate([[(pdb[ : 4], chain) for chain in pdb[5 : ].split('-')] for pdb in grouped_chains])
        biounits_chains = np.concatenate([biounit_chains_array[np.logical_and(pdbs_array == pdb, chains_array == chain)] for pdb, chain in split_chains])
        res_dict[node] = biounits_chains
    
    return res_dict


def find_repartition(chains_dict, homomers, heteromers):

    """
    Return a dictionary similar to the one created by find_chains_in_graph, with an additional level of classification for single chains, homomers and heteromers
    Dictionary structure : {'single_chains' : {cluster_name : [biounit chains]}, 'homomers' : {cluster_name : [biounit chains]}, 'heteromers' : {cluster_name : [biounit chains]}}
    Additionaly return the number of chains in each class (single chains, ...)
    """

    classes_dict = {'single_chains' : defaultdict(lambda : []), 'homomers' : defaultdict(lambda : []), 'heteromers' : defaultdict(lambda : [])}
    n_single_chains, n_homomers, n_heteromers = 0, 0, 0

    for node in chains_dict.keys():

        for k, chain in enumerate(chains_dict[node]):
            if tuple(chain) in homomers:
                classes_dict['homomers'][node].append(chain)
                n_homomers += 1
            elif tuple(chain) in heteromers:
                classes_dict['heteromers'][node].append(chain)
                n_heteromers += 1
            else:
                classes_dict['single_chains'][node].append(chain)
                n_single_chains += 1
    
    for node in classes_dict['single_chains']:
        classes_dict['single_chains'][node] = np.stack(classes_dict['single_chains'][node])
    
    for node in classes_dict['homomers']:
        classes_dict['homomers'][node] = np.stack(classes_dict['homomers'][node])
    
    for node in classes_dict['heteromers']:
        classes_dict['heteromers'][node] = np.stack(classes_dict['heteromers'][node])
    
    return classes_dict, n_single_chains, n_homomers, n_heteromers


def find_subgraphs_infos(subgraphs, clusters_dict, biounit_chains_array, pdbs_array, chains_array, homomers, heteromers):

    """
    Given a list of subgraphs, return a list of dictionaries and an array of sizes of the same length.
    Dictionaries are the chains_dict and classes_dict corresponding to each subgraph, returned by the find_chains_in_graph and find_repartition functions respectively
    The array of sizes is of shape (len(subgraph), 3). It gives the number of single chains, homomers and heteromers present in each subgraph
    """

    size_array = np.zeros((len(subgraphs), 3))
    dict_list = []
    for k, subgraph in tqdm(enumerate(subgraphs)):
        chains_dict = find_chains_in_graph(subgraph, clusters_dict, biounit_chains_array, pdbs_array, chains_array)
        classes_dict, n_single_chains, n_homomers, n_heteromers = find_repartition(chains_dict, homomers, heteromers)
        size_array[k] = np.array([n_single_chains, n_homomers, n_heteromers])
        dict_list.append((chains_dict, classes_dict))
    
    return dict_list, size_array, int(np.sum(size_array[:, 0])), int(np.sum(size_array[:, 1])), int(np.sum(size_array[:, 2]))


def construct_dataset(dict_list, size_array, indices):

    """
    Given the dict_list and size_array returned by find_subgraphs_info, return the 2 dictionaries (chains_dict and classes_dict) corresponding to the graph encompassing all the subgraphs indicated by indices
    Additionally return the number of single chains, homomers and heteromers in this supergraph 
    """

    dataset_clusters_dict = {}
    dataset_classes_dict = {'single_chains' : {}, 'homomers' : {}, 'heteromers' : {}}
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
    
    return dataset_clusters_dict, dataset_classes_dict, single_chains_size, homomers_size, heteromers_size


def remove_elements_from_dataset(indices, remaining_indices, chain_class, size_obj, current_sizes, size_array, tolerance=.2):

    """
    Remove values from indices untill we get the required (size_obj) number of chains in the class of interest (chain_class)
    Parameter chain_class corresponds to the single chain (0), homomer (1) or heteromer (2) class
    """

    sizes = [s[chain_class] for s in size_array[indices]]
    sorted_sizes_indices = np.argsort(sizes)[::-1]

    while current_sizes[chain_class] > size_obj and len(sorted_sizes_indices) > 0:

        if current_sizes[chain_class] - size_array[indices[sorted_sizes_indices[0]], chain_class] < (1 - tolerance) * size_obj:
            sorted_sizes_indices = sorted_sizes_indices[1 : ]
            continue
        
        current_sizes -= size_array[indices[sorted_sizes_indices[0]]]
        remaining_indices.append(indices[sorted_sizes_indices[0]])
        indices.pop(sorted_sizes_indices[0])
        sorted_sizes_indices = sorted_sizes_indices[1 : ] if len(sorted_sizes_indices) > 0 else []
    
    return indices, remaining_indices, current_sizes[0], current_sizes[1], current_sizes[2]


def add_elements_to_dataset(indices, remaining_indices, chain_class, size_obj, current_sizes, size_array, tolerance=.2):

    """
    Add values to indices untill we get the required (size_obj) number of chains in the class of interest (chain_class)
    Parameter chain_class corresponds to the single chain (0), homomer (1) or heteromer (2) class
    """

    sizes = [s[chain_class] for s in size_array[remaining_indices]]
    sorted_sizes_indices = np.argsort(sizes)[::-1]

    while current_sizes[chain_class] < size_obj and len(sorted_sizes_indices) > 0:

        if current_sizes[chain_class] + size_array[remaining_indices[sorted_sizes_indices[0]], chain_class] > (1 + tolerance) * size_obj:
            sorted_sizes_indices = sorted_sizes_indices[1 : ]
            continue

        current_sizes += size_array[remaining_indices[sorted_sizes_indices[0]]]
        indices.append(remaining_indices[sorted_sizes_indices[0]])
        remaining_indices.pop(sorted_sizes_indices[0])
        sorted_sizes_indices = sorted_sizes_indices[1 : ] if len(sorted_sizes_indices) > 0 else []
    
    return indices, remaining_indices, current_sizes[0], current_sizes[1], current_sizes[2]


def adjust_dataset(indices, remaining_indices, dict_list, size_array, n_single_chains, n_homomers, n_heteromers, single_chains_size, homomers_size, heteromers_size, tolerance=.2):

    """
    If required, remove and add values in indices so that the number of chains in each class correspond to the required numbers within a tolerance.
    First remove and then add (if necessary, for each class separately).
    In the end, we might end up with more chains than desired in the first 2 classes but for a reasonable tolerance (~10-20 %), this should not happen.
    """

    if single_chains_size > (1 + tolerance) * n_single_chains:
        (
            indices,
            remaining_indices,
            single_chains_size,
            homomers_size,
            heteromers_size,
        ) = remove_elements_from_dataset(indices, remaining_indices, 0, n_single_chains, np.array([single_chains_size, homomers_size, heteromers_size]), size_array, tolerance=tolerance)
    
    if homomers_size > (1 + tolerance) * n_homomers:
        (
            indices,
            remaining_indices,
            single_chains_size,
            homomers_size,
            heteromers_size,
        ) = remove_elements_from_dataset(indices, remaining_indices, 1, n_homomers, np.array([single_chains_size, homomers_size, heteromers_size]), size_array, tolerance=tolerance)

    if heteromers_size > (1 + tolerance) * n_heteromers:
        (
            indices,
            remaining_indices,
            single_chains_size,
            homomers_size,
            heteromers_size,
        ) = remove_elements_from_dataset(indices, remaining_indices, 2, n_heteromers, np.array([single_chains_size, homomers_size, heteromers_size]), size_array, tolerance=tolerance)
    
    if single_chains_size < (1 - tolerance) * n_single_chains:
        (
            indices,
            remaining_indices,
            single_chains_size,
            homomers_size,
            heteromers_size,
        ) = add_elements_to_dataset(indices, remaining_indices, 0, n_single_chains, np.array([single_chains_size, homomers_size, heteromers_size]), size_array, tolerance=tolerance)

    if homomers_size < (1 - tolerance) * n_homomers:
        (
            indices,
            remaining_indices,
            single_chains_size,
            homomers_size,
            heteromers_size,
        ) = add_elements_to_dataset(indices, remaining_indices, 1, n_homomers, np.array([single_chains_size, homomers_size, heteromers_size]), size_array, tolerance=tolerance)

    if heteromers_size < (1 - tolerance) * n_heteromers:
        (
            indices,
            remaining_indices,
            single_chains_size,
            homomers_size,
            heteromers_size,
        ) = add_elements_to_dataset(indices, remaining_indices, 2, n_heteromers, np.array([single_chains_size, homomers_size, heteromers_size]), size_array, tolerance=tolerance)

    (
        dataset_clusters_dict,
        dataset_classes_dict,
        single_chains_size,
        homomers_size,
        heteromers_size,
    ) = construct_dataset(dict_list, size_array, indices)
    return dataset_clusters_dict, dataset_classes_dict, single_chains_size, homomers_size, heteromers_size, remaining_indices


def fill_dataset(dict_list, size_array, n_samples, n_single_chains, n_homomers, n_heteromers, remaining_indices, n_max_iter=50, tolerance=.2):
    
    """
    Given a list of indices to choose from (remaining_indices), choose a list of subgraphs to construct a dataset containing the required number of biounits for each class (single chains, ...) within a tolerance.
    Return the same outputs as the construct_dataset function, as long as the list of remaining indices after selection.
    """

    single_chains_size, homomers_size, heteromers_size = 0, 0, 0
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
        ) = construct_dataset(dict_list, size_array, indices)
        distribution_satisfied = single_chains_size > (1 - tolerance) * n_single_chains and single_chains_size < (1 + tolerance) * n_single_chains and homomers_size > (1 - tolerance) * n_homomers and homomers_size < (1 + tolerance) * n_homomers and heteromers_size > (1 - tolerance) * n_heteromers and heteromers_size < (1 + tolerance) * n_heteromers
    
    if not distribution_satisfied:
        (
            dataset_clusters_dict,
            dataset_classes_dict,
            single_chains_size,
            homomers_size,
            heteromers_size,
            remaining_indices,
        ) = adjust_dataset(
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
                            tolerance=tolerance,
                            )
    else:
        remaining_indices = [i for i in remaining_indices if i not in indices]
    
    print("Number of samplings (fill_dataset):", n_iter)
    return dataset_clusters_dict, dataset_classes_dict, remaining_indices, single_chains_size, homomers_size, heteromers_size


def split_dataset(graph, clusters_dict, merged_seqs_dict, dataset_dir, valid_split=.05, test_split=.05, tolerance=.2):

    """
    Given a graph representing connections between MMSeqs2 clusters, split the dataset between train, validation and test

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
    valid_split : float in ]0, 1[, default 0.05
        the validation split ratio
    test_split : float in ]0, 1[, default 0.05
        the test split ratio
    
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
    single_chains : list of str
        the list of all biounit chains that are in a single chain state (in their biounit)
    homomers : list of str
        the list of all biounit chains that are in a homomeric state (in their biounit)
    heteromers : list of str
        the list of all biounit chains that are in a heteromeric state (in their biounit)
    """

    subgraphs = np.array([graph.subgraph(c) for c in sorted(nx.connected_components(graph), key=len, reverse=True)], dtype=object)
    remaining_indices = list(np.arange(1, len(subgraphs)))
    seqs_names_list = retrieve_seqs_names_list(merged_seqs_dict)
    pdb_seqs_dict = create_pdb_seqs_dict(seqs_names_list)
    single_chains, homomers, heteromers = divide_according_to_chains_interactions(pdb_seqs_dict, dataset_dir)
    biounit_chains_array = np.array(single_chains + homomers + heteromers)
    pdbs_array = np.array([c[0][ : 4] for c in biounit_chains_array])
    chains_array = np.array([c[1] for c in biounit_chains_array])

    (
        dict_list,
        size_array,
        n_single_chains,
        n_homomers,
        n_heteromers
    ) = find_subgraphs_infos(
                                subgraphs,
                                clusters_dict,
                                biounit_chains_array,
                                pdbs_array,
                                chains_array,
                                homomers,
                                heteromers,
                            )
    
    n_single_chains_valid, n_homomers_valid, n_heteromers_valid = valid_split * np.array([n_single_chains, n_homomers, n_heteromers])
    n_single_chains_test, n_homomers_test, n_heteromers_test = test_split * np.array([n_single_chains, n_homomers, n_heteromers])
    n_samples_valid, n_samples_test = int(valid_split * len(subgraphs)), int(test_split * len(subgraphs))

    (
        valid_clusters_dict,
        valid_classes_dict,
        remaining_indices,
        n_single_chains_valid,
        n_homomers_valid,
        n_heteromers_valid,
    ) = fill_dataset(
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
    ) = fill_dataset(
                        dict_list,
                        size_array,
                        n_samples_test,
                        n_single_chains_test,
                        n_homomers_test,
                        n_heteromers_test,
                        remaining_indices,
                        tolerance=tolerance,
                    )
    
    remaining_indices.append(0) # add the big first cluster, that we always want in the training set
    (
        train_clusters_dict,
        train_classes_dict,
        n_single_chains_train,
        n_homomers_train,
        n_heteromers_train,
    ) = construct_dataset(dict_list, size_array, remaining_indices)

    print("Classes distribution (single chain / homomer / heteromer):")
    print("Train set:", int(n_single_chains_train), '/', int(n_homomers_train), '/', int(n_heteromers_train))
    print("Validation set:", int(n_single_chains_valid), '/', int(n_homomers_valid), '/', int(n_heteromers_valid))
    print("Test set:", int(n_single_chains_test), '/', int(n_homomers_test), '/', int(n_heteromers_test))

    return train_clusters_dict, train_classes_dict, valid_clusters_dict, valid_classes_dict, test_clusters_dict, test_classes_dict, single_chains, homomers, heteromers


def build_dataset_partition(dataset_dir, tmp_folder, valid_split=.05, test_split=.05, tolerance=.2):

    """
    Build training, validation and test sets from a curated dataset of biounit, using MMSeqs2 for clustering.

    Parameters
    ----------
    dataset_dir : str
        the path to the dataset
    valid_split : float in ]0, 1[, default 0.05
        the validation split ratio
    test_split : float in ]0, 1[, default 0.05
        the test split ratio
    
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

    # retrieve all sequences and create a merged_seqs_dict
    seqs_dict = load_pdbs(dataset_dir)
    merged_seqs_dict = merge_chains(seqs_dict)

    # write sequences to a fasta file for clustering with MMSeqs2, run MMSeqs2 and delete the fasta file
    fasta_file = os.path.join(tmp_folder, 'all_seqs.fasta')
    write_fasta(fasta_file, merged_seqs_dict)
    run_mmseqs2(fasta_file, tmp_folder)
    subprocess.run(['rm', fasta_file])

    # retrieve MMSeqs2 clusters and build a graph with these clusters
    clusters_dict, clusters_pdb_dict = read_clusters(os.path.join('MMSeqs2_results/clusterRes_all_seqs.fasta'))
    subprocess.run(['rm', '-r', os.path.join(tmp_folder, 'MMSeqs2_results')])
    graph = make_graph(clusters_pdb_dict)

    # perform the splitting into train, validation and tesst sets
    (
        train_clusters_dict,
        train_classes_dict,
        valid_clusters_dict,
        valid_classes_dict,
        test_clusters_dict,
        test_classes_dict,
        _, _, _,
    ) = split_dataset(graph, clusters_dict, merged_seqs_dict, dataset_dir, valid_split=valid_split, test_split=test_split, tolerance=tolerance)

    return train_clusters_dict, train_classes_dict, valid_clusters_dict, valid_classes_dict, test_clusters_dict, test_classes_dict