from email.policy import default
import editdistance
from collections import defaultdict
import numpy as np
from tqdm import tqdm


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
    print('Searching for duplicates...')
    for key in tqdm(pdb_ids):
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
    print('Calculating sequence similarities...')
    for key in tqdm(pdb_ids):
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
    
def get_identical_sequence_weird_chains(data_scn, pdb_ids=None, similarity_thr=0.9, dataset="train"):
    """
    Get a dictionary of recombined weird chain lists that have a high similarity to a main chain
    """

    sequence_ids = get_weird_chain_sequence_id(data_scn, pdb_ids, dataset)
    res = defaultdict(lambda: {})
    print('Finding chains to combine...')
    for key, id_dict in tqdm(sequence_ids.items()):
        for chain_id, value in id_dict.items():
            if value["similarity"] > similarity_thr:
                res[key][chain_id] = value["chains"]
    return res

def get_minus_weird_chains(data_scn, pdb_ids=None, dataset="train"):
    """
    Get a dictionary of shortened weird chains (repeated for duplicates and unique)
    """

    key_dict, pdb_ids = get_keys_ids(data_scn, pdb_ids, dataset)
    res = {}
    print('Searching for shortened chains...')
    for key in tqdm(pdb_ids):
        main_chains = [x for x in key_dict[key] if len(x.split('_')[-1]) <= 2]
        main_chain_ids = [x.split('_')[-1].lower() for x in main_chains]
        minus_chains = [x for x in key_dict[key] if x.split("_")[-1] == "-"]
        if len(minus_chains) > 0:
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

def filter_chain_pars(
        data_scn, 
        dataset="train", 
        resolution_thr=3.5,
        missing_thr=0.9,
        length_thr=30,
    ):
    key_dict, _ = get_keys_ids(data_scn, dataset=dataset)
    result = defaultdict(lambda: [])
    print('Filtering by resolution, length and missing values...')
    for i, res in tqdm(enumerate(data_scn[dataset]["res"]), total=len(data_scn[dataset]["res"])):
        key = data_scn[dataset]["ids"][i].split('_')[0]
        if res is None or res > resolution_thr:
            result[key] = key_dict[data_scn[dataset]["ids"][i].split('_')[0]]
            continue
        nm = (np.array(list(data_scn[dataset]["msk"][i])) == "+").sum()
        total_len = len(data_scn[dataset]["msk"][i])
        if nm < length_thr:
            result[key].append(data_scn[dataset]["ids"][i])
        elif nm / total_len < missing_thr:
            result[key] = key_dict[data_scn[dataset]["ids"][i].split('_')[0]]
    return result

def filter_weird_chains(data_scn, dataset, similarity_thr=0.9, resolution_thr=3.5, length_thr=30, missing_thr=0.9):
    """
    Generate lists of chains to remove, combine and keep (in this order)
    """

    key_dict, _ = get_keys_ids(data_scn, dataset)
    minus_chain_dict = get_minus_weird_chains(data_scn, dataset=dataset)
    key_dict = shorten_key_dict(key_dict, minus_chain_dict)
    dup_chain_dict = get_duplicate_weird_chains(data_scn, pdb_ids=list(key_dict.keys()), dataset=dataset)
    key_dict = shorten_key_dict(key_dict, dup_chain_dict)
    id_seq_chain_dict = get_identical_sequence_weird_chains(data_scn, pdb_ids=list(key_dict.keys()), similarity_thr=similarity_thr, dataset=dataset)
    key_dict = shorten_key_dict(key_dict, id_seq_chain_dict)
    res_dict = filter_chain_pars(
        data_scn,
        dataset, 
        resolution_thr=resolution_thr,
        length_thr=length_thr,
        missing_thr=missing_thr,
    )
    to_remove = []
    for value in key_dict.values():
        to_remove += value
    for value in res_dict.values():
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
    print(f'Adding {len(to_add)} chains...')
    for i, key in enumerate(add_data_scn[dataset]["ids"]):
        if key in to_add:
            for x in main_data_scn[dataset]:
                main_data_scn[dataset][x].append(add_data_scn[dataset][x][i])
    return main_data_scn

def preprocess(data_scn, dataset="train", similarity_thr=0.9, resolution_thr=3.5, length_thr=30, missing_thr=0.1):
    """
    Filter weird chains and combine chains with the same PDB ID into single entries
    """
    
    to_remove, to_combine, to_keep = filter_weird_chains(
        data_scn, 
        dataset=dataset, 
        similarity_thr=similarity_thr,
        resolution_thr=resolution_thr,
        length_thr=length_thr,
        missing_thr=missing_thr,
    )
    print(f'Removing {len(to_remove)} chains... \n')
    print(f'Recombining chains...')
    to_keep_pdbs = defaultdict(lambda: [])
    for key in to_keep:
        to_keep_pdbs[key.split('_')[0]].append(key)
    new_data = defaultdict(lambda: [])
    for chain_list in tqdm(to_combine):
        chain_id = chain_list[0][10].upper()
        pdb_id = chain_list[0].split('_')[0]
        chain_indices = [data_scn[dataset]["ids"].index(x) for x in chain_list]
        new_chain = f"{pdb_id}_r_{chain_id}"
        new_seq = ''.join([data_scn[dataset]["seq"][i] for i in chain_indices])
        new_msk = ''.join([data_scn[dataset]["msk"][i] for i in chain_indices])
        new_crd = np.concatenate([data_scn[dataset]["crd"][i] for i in chain_indices])
        data_scn[dataset]["ids"].append(new_chain)
        data_scn[dataset]["seq"].append(new_seq)
        data_scn[dataset]["crd"].append(new_crd)
        data_scn[dataset]["msk"].append(new_msk)
        to_keep_pdbs[pdb_id].append(new_chain)
    print(f'Generating multi-chain entries...')
    for key in tqdm(to_keep_pdbs):
        chain_list = to_keep_pdbs[key]
        chain_indices = [data_scn[dataset]["ids"].index(x) for x in sorted(chain_list)]
        new_seq = [data_scn[dataset]["seq"][i] for i in chain_indices]
        new_crd = [data_scn[dataset]["crd"][i] for i in chain_indices]
        new_msk = [data_scn[dataset]["msk"][i] for i in chain_indices]
        new_data["ids"].append(key)
        new_data["scn"].append(sorted(chain_list))
        new_data["seq"].append(new_seq)
        new_data["crd"].append(new_crd)
        new_data["msk"].append(new_msk)
    return new_data

def cut_missing_ends(data):
    """
    Cut intervals at the end and start of sequences where structure information is missing
    """

    cut_start = 0
    start_lens = []
    cut_end = 0
    end_lens = []
    print('Cutting missing ends...')
    for i in tqdm(range(len(data["ids"]))):
        for j, mask in enumerate(data["msk"][i]):
            k = 0
            while mask[k] == "-":
                k += 1
            if k > 0:
                cut_start += 1
                start_lens.append(k)
                data["msk"][i][j] = data["msk"][i][j][: k]
                data["crd"][i][j] = data["crd"][i][j][: k * 14]
                data["seq"][i][j] = data["seq"][i][j][: k]
            k = len(data["msk"][i][j]) - 1
            while mask[k] == "-":
                k -= 1
            k += 1
            if k < len(data["msk"][i][j]) - 1:
                cut_end += 1
                end_lens.append(len(data["msk"][i][j]) - 1 - k)
                data["msk"][i][j] = data["msk"][i][j][k:]
                data["crd"][i][j] = data["crd"][i][j][k * 14:]
                data["seq"][i][j] = data["seq"][i][j][k:]
    print(f'Cut {cut_start} start intervals (mean length {np.array(start_lens).mean():.1f}) and {cut_end} end intervals (mean length {np.array(end_lens).mean():.1f})')
    return data