import editdistance
from collections import defaultdict
import numpy as np


def get_keys_ids(data_scn, pdb_ids=None, dataset="train"):
    """
    Get a PDB ID to chain list correspondence + set the pdb_ids value to all PDB IDs if it's None
    """

    key_dict = defaultdict(lambda: [])
    for key in data_scn[dataset]["ids"]:
        key_dict[key.split('_')[0]].append(key)
    if pdb_ids is None:
        pdb_ids = list(key_dict.keys())
    return key_dict, pdb_ids

def get_duplicate_weird_chains(data_scn, pdb_ids=None, dataset="train"):
    """
    Get a dictionary with duplicate chains (keys are PDB ids, values are lists of redundant chains)
    """

    key_dict, pdb_ids = get_keys_ids(data_scn, pdb_ids, dataset)
    res = defaultdict(lambda: [])
    for key in pdb_ids:
        chains = key_dict[key]
        main_chains = [x.split('_')[-1].lower() for x in chains if len(x.split('_')[-1]) <= 2]
        weird_chains = [x for x in chains if len(x.split('_')[-1]) > 2]
        for chain in weird_chains:
            assert len(chain) == 12
            if chain[10].lower() in main_chains and chain[-1] != '-':
                res[key].append(chain)
    return res

def get_weird_chain_sequence_id(data_scn, pdb_ids=None, dataset="train"):
    """
    Get a dictionary with highest main chain similarities of recombined weird chains 
    """

    key_dict, pdb_ids = get_keys_ids(data_scn, pdb_ids, dataset)
    sequence_ids = defaultdict(lambda: {})
    for key in pdb_ids:
        chains = key_dict[key]
        main_chains = [x for x in chains if len(x.split('_')[-1]) <= 2]
        weird_chains = [x for x in chains if len(x.split('_')[-1]) > 2]
        weird_chain_ids = [x[10].lower() for x in weird_chains]
        main_chain_ids = [x.split('_')[-1].lower() for x in main_chains]
        if len(main_chains) < 1 or len(weird_chains) < 1:
            continue
        for weird_chain_id in weird_chain_ids:
            if weird_chain_id in main_chain_ids:
                continue
            metrics = []
            seq = ''
            chain_list = sorted([x for x in weird_chains if x[10].lower() == weird_chain_id and x[-1] != '-'])
            for chain in chain_list:
                n = data_scn[dataset]["ids"].index(chain)
                seq += data_scn[dataset]["seq"][n]
            for main_chain in main_chains:
                n = data_scn[dataset]["ids"].index(main_chains[0])
                full_seq = data_scn[dataset]["seq"][n]
                l = max(len(full_seq), len(seq))
                metrics.append((l - editdistance.eval(full_seq, seq)) / l)
            sequence_ids[key][weird_chain_id] = {"similarity": max(metrics), "chains": chain_list}
    return sequence_ids
    
def get_identical_sequence_weird_chains(data_scn, pdb_ids=None, thr=0.9, dataset="train"):
    """
    Get a dictionary of recombined weird chain lists that have a high similarity to a main chain
    """

    sequence_ids = get_weird_chain_sequence_id(data_scn, pdb_ids, dataset)
    res = defaultdict(lambda: {})
    for key, id_dict in sequence_ids.items():
        for chain_id, value in id_dict.items():
            if value["similarity"] > thr:
                res[key][chain_id] = value["chains"]
    return res

def get_minus_weird_chains(data_scn, pdb_ids=None, dataset="train"):
    """
    Get a dictionary of shortened weird chains (repeated for duplicates and unique)
    """

    key_dict, pdb_ids = get_keys_ids(data_scn, pdb_ids, dataset)
    res = {}
    for key in pdb_ids:
        main_chains = [x for x in key_dict[key] if len(x.split('_')[-1]) <= 2]
        main_chain_ids = [x.split('_')[-1].lower() for x in main_chains]
        minus_chains = [x for x in key_dict[key] if len(x.split("_")[-1]) > 2]
        res[key] = {
            "repeated": [x for x in minus_chains if x[10].lower() in main_chain_ids],
            "unique": [x for x in minus_chains if x[10].lower() not in main_chain_ids]
        }
    return res

def shorten_key_dict(key_dict, other_dict):
    """
    Remove explained chains from the dictionary
    """

    for key, value in other_dict.items():
        if isinstance(value, dict):
            key_dict[key] = [x for x in key_dict[key] if all([x not in y for y in value.values() if isinstance(y, list)])]
        else:
            key_dict[key] = [x for x in key_dict[key] if x not in value]
        if len(key_dict[key]) == 0:
            key_dict.pop(key)
    key_dict = filter_main_chain(key_dict)
    return key_dict

def filter_main_chain(key_dict):
    """
    Remove PDB ids with only main chains from the dictionary
    """

    for key in list(key_dict.keys()):
        if all([len(x.split('_')[-1]) <= 2 for x in key_dict[key]]):
            key_dict.pop(key)
    return key_dict

def filter_weird_chains(data_scn, dataset, thr=0.9):
    """
    Generate lists of chains to remove, combine and keep (in this order)
    """

    key_dict, _ = get_keys_ids(data_scn, dataset)
    minus_chain_dict = get_minus_weird_chains(data_scn, dataset)
    key_dict = shorten_key_dict(key_dict, minus_chain_dict)
    dup_chain_dict = get_duplicate_weird_chains(data_scn, pdb_ids=list(key_dict.keys()), dataset=dataset)
    key_dict = shorten_key_dict(key_dict, dup_chain_dict)
    id_seq_chain_dict = get_identical_sequence_weird_chains(data_scn, pdb_ids=list(key_dict.keys()), thr=thr, dataset=dataset)
    key_dict = shorten_key_dict(key_dict, id_seq_chain_dict)
    to_remove = []
    for value in key_dict.values():
        to_remove += value
    for value in minus_chain_dict.values():
        to_remove += value["repeated"]
    for value in dup_chain_dict.values():
        to_remove += value
    to_combine = []
    all_to_combine = []
    for value in id_seq_chain_dict.values():
        for chain_list in value.values():
            all_to_combine += chain_list
            to_combine.append(chain_list)
    to_keep = [x for x in data_scn[dataset]["ids"] if x not in to_remove and x not in all_to_combine]
    return to_remove, to_combine, to_keep

def combine_data(main_data_scn, add_data_scn, dataset="train"):
    """
    Expand `main_data_scn` with chains from `add_data_scn` that share the PDB entry with chains in `main_data_scn`
    """

    main_key_dict, _ = get_keys_ids(main_data_scn, dataset=dataset)
    add_key_dict, _ = get_keys_ids(add_data_scn, dataset=dataset)
    to_add = []
    for key, value in main_key_dict.items():
        for chain in add_key_dict[key]:
            if chain not in value:
                to_add.append(chain)
    for i, key in enumerate(add_data_scn[dataset]["ids"]):
        if key in to_add:
            for x in main_data_scn[dataset]:
                main_data_scn[dataset][x].append(add_data_scn[dataset][x][i])
    return main_data_scn

def preprocess(data_scn, dataset="train", thr=0.9):
    """
    Filter weird chains and combine chains with the same PDB ID into single entries
    """
    
    _, to_combine, to_keep = filter_weird_chains(data_scn, dataset=dataset, thr=thr)
    to_keep_pdbs = defaultdict(lambda: [])
    for key in to_keep:
        to_keep_pdbs[key.split('_')[0]].append(key)
    new_data = defaultdict(lambda: [])
    for chain_list in to_combine:
        chain_id = chain_list[0][10].upper()
        pdb_id = chain_list[0].split('_')[0]
        chain_indices = [data_scn[dataset]["ids"].index(x) for x in chain_list]
        new_chain = f"{pdb_id}_r_{chain_id}"
        new_seq = sum([data_scn[dataset]["seq"][i] for i in chain_indices])
        new_crd = np.concatenate([data_scn["seq"][i] for i in chain_indices])
        data_scn[dataset]["ids"].append(new_chain)
        data_scn[dataset]["seq"].append(new_seq)
        data_scn[dataset]["crd"].append(new_crd)
        to_keep_pdbs[pdb_id].append(new_chain)

    for key in to_keep_pdbs:
        chain_list = to_keep_pdbs[key]
        chain_indices = [data_scn[dataset]["ids"].index(x) for x in sorted(chain_list)]
        new_seq = [data_scn[dataset]["seq"][i] for i in chain_indices]
        new_crd = [data_scn["seq"][i] for i in chain_indices]
        new_data["ids"].append(key)
        new_data["scn"].append(sorted(chain_list))
        new_data["seq"].append(new_seq)
        new_data["crd"].append(new_crd)
    return new_data