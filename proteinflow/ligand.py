"""Methods for retrieving ligand information from PDB files and clustering them using Tanimoto similarity."""
import warnings

import numpy as np
import pypdb
from Bio import BiopythonWarning
from Bio.PDB import PDBParser
from tqdm import tqdm

with warnings.catch_warnings():
    warnings.simplefilter("ignore", BiopythonWarning)

import copy
import gzip
import mmap
import os
import pickle
import shutil
from collections import defaultdict
from functools import partial

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdFingerprintGenerator
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from rdkit.ML.Cluster import Butina

D3TO1 = {
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


def is_ion(st):
    """Determine if a string is an ion."""
    if len("".join([i for i in st if not i.isdigit()])) <= 2:
        return True
    if st in D3TO1.keys():
        return True
    return False


def atom2mol(p):
    """Convert a PDB file to a molecule."""
    small_molecules = p.df["HETATM"]

    # Group small molecules using residue3to1()
    small_molecules["residue_name"] = small_molecules["residue_name"].apply(
        lambda x: x.upper()
    )
    small_molecules["residue_number"] = small_molecules["residue_number"].astype(int)
    small_molecules["chain_id"] = small_molecules["chain_id"].astype(str)
    small_molecules["residue"] = small_molecules.apply(
        lambda row: row["residue_name"] + str(row["residue_number"]) + row["chain_id"],
        axis=1,
    )

    grouped_small_molecules = small_molecules.groupby("residue").first()
    return grouped_small_molecules[["chain_id", "residue_name"]]


def get_covalent_bonds(
    chain1, chain2, connect_dict, distance_check=False, Time_profiling=False
):
    """Get covalent bonds between two chains."""
    from time import time

    t0 = time()
    atoms_chain1 = []
    # atom2M_1 = {}

    atom2M_1_num = {}
    atom2M_1_num_reverse = {}
    for i in range(len(chain1)):
        atoms_chain1 += list(chain1[i].get_atoms())
        # atom2M_1[chain1[i].get_id()[1]] = list(chain1[i].get_atoms())

        atom2M_1_num[chain1[i].get_id()[1]] = []
        for at in list(chain1[i].get_atoms()):
            atom2M_1_num[chain1[i].get_id()[1]].append(at.get_serial_number())
            if (
                at.get_serial_number() in connect_dict
                or at.get_serial_number() in connect_dict.values()
            ):
                atom2M_1_num_reverse[at.get_serial_number()] = chain1[i].get_id()[1]

    t1 = time()

    atoms_chain2 = []
    # atom2M_2 = {}
    atom2M_2_num = {}
    atom2M_2_num_reverse = {}
    for i in range(len(chain2)):
        atoms_chain2 += list(chain2[i].get_atoms())
        # atom2M_2[chain2[i].get_id()[1]] = list(chain2[i].get_atoms())

        atom2M_2_num[chain2[i].get_id()[1]] = []
        for at in list(chain2[i].get_atoms()):
            atom2M_2_num[chain2[i].get_id()[1]].append(at.get_serial_number())
            if (
                at.get_serial_number() in connect_dict
                or at.get_serial_number() in connect_dict.values()
            ):
                atom2M_2_num_reverse[at.get_serial_number()] = chain2[i].get_id()[1]

    t2 = time()
    # Check for covalent bonds between the chains
    covalent_bonds_new = []

    for atom1 in atom2M_1_num_reverse:
        for atom2 in atom2M_2_num_reverse:
            if (
                atom2 in atom2M_1_num[atom2M_1_num_reverse[atom1]]
                or atom1 in atom2M_2_num[atom2M_2_num_reverse[atom2]]
            ):
                continue
            if (atom1 in connect_dict and atom2 in connect_dict[atom1]) or (
                atom2 in connect_dict and atom1 in connect_dict[atom2]
            ):
                covalent_bonds_new.append((atom1, atom2))
    t3 = time()
    if Time_profiling:
        total_time = t3 - t0
        print("Get atom chains 1 ratio: ", (t1 - t0) / total_time)
        print("Get atom chains 2 ratio: ", (t2 - t1) / total_time)
        print("Get covalent bonds ratio: ", (t3 - t2) / total_time)
        print(len(atoms_chain1), len(atoms_chain2), len(covalent_bonds_new))
        print("Total time: ", total_time)
    return covalent_bonds_new, None, None


def fix_connect(conect_line, minimum=0):
    """Fix a CONECT line."""
    if "CONECT" not in conect_line:
        return ""
    separated_line = conect_line.split(" ")
    separated_line = list(filter(("").__ne__, separated_line))
    try:
        separated_line.remove("\n")
    except Exception:
        pass
    conect_line = " ".join(separated_line)
    if len(separated_line) >= 3:
        return conect_line
    if len(separated_line) == 1 or separated_line[0] != "CONECT":
        conect_line = conect_line.replace("CONECT", "CONECT ")
    separated_line = conect_line.split(" ")
    if len(separated_line) == 2:
        numbers = separated_line[1]
        potential_connections = 10
        diffs = []
        numbers_lists = []
        for i in range(2, potential_connections):
            if len(numbers) % i == 0:
                numbers_list = [
                    numbers[j : j + len(numbers) // i]
                    for j in range(0, len(numbers), len(numbers) // i)
                ]
                # minimum difference between every numbers pair
                mean_diff = 0
                all_small = 1
                for k in range(len(numbers_list)):
                    if int(numbers_list[k]) >= minimum:
                        all_small *= 0
                    for l_idx in range(k + 1, len(numbers_list)):
                        mean_diff += abs(
                            int(numbers_list[k]) - int(numbers_list[l_idx])
                        )
                mean_diff /= len(numbers_list)
                le = len(numbers) // i
                if all_small:
                    mean_diff += 100000
                diffs.append(mean_diff / (le * le * le))
                numbers_lists.append(numbers_list)
        diffs = np.array(diffs)
        if len(diffs) > 0:
            best_candidates = diffs[diffs == min(diffs)]
            if best_candidates.shape[0] > 1:
                # choose candidate with highest numbers list length
                best_candidates_lengths = list(
                    [len(numbers_lists[i]) for i in (diffs == min(diffs)).nonzero()[0]]
                )
                best_candidate = best_candidates_lengths.index(
                    max(best_candidates_lengths)
                )
            else:
                best_candidate = (diffs == min(diffs)).nonzero()[0][0]
            conect_line = "CONECT " + " ".join(numbers_lists[best_candidate])
    return conect_line


def parse_pdb_file(pdb_file, minimum=0):
    """Parse a PDB file and return a dictionary with the connectivity information."""
    connect_dict = {}
    with open(pdb_file) as f:
        for line in f:
            if line.startswith("CONECT"):
                line = fix_connect(line, minimum=minimum)
                try:
                    fields = line.split()
                    atom_serial = int(fields[1])
                    connected_atoms = [int(x) for x in fields[2:] if x.isdigit()]
                    connect_dict[atom_serial] = connected_atoms
                except Exception:
                    pass
    return connect_dict


def connected(claster1, cluster2, connectivity):
    """Check if two clusters are connected."""
    for atom1 in claster1:
        for atom2 in cluster2:
            if connectivity[atom1, atom2] == 1:
                return True
    return False


def merge_components(clusters, connectivity):
    """Merge clusters that are connected."""
    if len(clusters) == 1:
        return clusters
    for i, cluster1 in enumerate(clusters):
        for j, cluster2 in enumerate(clusters):
            if cluster1 == cluster2:
                continue
            if connected(cluster1, cluster2, connectivity):
                clusters[i] = cluster1 + cluster2
                clusters.remove(cluster2)
                return merge_components(clusters, connectivity)
    return clusters


def get_raw_ligands(file_path, ligand):
    """Get the raw ligands from a PDB file."""
    ligands = []
    for molecule in ligand:
        resname = molecule.getResname()
        ligand_id = f"{molecule.getChid()} {molecule.getResnum()}"
        is_ligand = False
        with open(file_path) as pdb_file:
            for line in pdb_file:
                if line.startswith("HET   " + resname + ligand_id) and "LIGAND" in line:
                    is_ligand = True
                    break
        if is_ligand:
            ligands.append(molecule)
    return ligands


def describe_chemical(chem_id):
    """
    Get the chemical description from pypdb.

    :param chem_id: chemical id
    :return: chemical description
    """
    if len(chem_id) > 3:
        raise Exception("Ligand id with more than 3 characters provided")

    return pypdb.get_info(
        chem_id, url_root="https://data.rcsb.org/rest/v1/core/chemcomp/"
    )


def process_ligand(pdb_string, res_name, tmp_pdb_string=None):
    """
    Add bond orders to a pdb ligand.

    1. Select the ligand component with name "res_name"
    2. Get the corresponding SMILES from pypdb
    3. Create a template molecule from the SMILES in step 2
    4. Write the PDB file to a stream
    5. Read the stream into an RDKit molecule
    6. Assign the bond orders from the template from step 3

    :param pdb_string: pdb string to process
    :param res_name: residue name of ligand to extract
    :param tmp_pdb_string: optional pdb string to use if the first one fails
    :return: molecule with bond orders assigned
    """
    # output = StringIO()
    # sub_mol = ligand.select(f"resname {res_name}")
    # sub_mol = ligand
    chem_desc = describe_chemical(f"{res_name}")
    sub_smiles = chem_desc["rcsb_chem_comp_descriptor"]["smiles"]
    template = AllChem.MolFromSmiles(sub_smiles)
    # writePDBStream(output, sub_mol)
    # pdb_string = output.getvalue()
    rd_mol = AllChem.MolFromPDBBlock(pdb_string)
    if tmp_pdb_string is not None:
        if rd_mol is None or rd_mol.GetNumAtoms() == 0:
            rd_mol = AllChem.MolFromPDBBlock(tmp_pdb_string)
    # if template is None or rd_mol is None:
    #    return template
    try:
        new_mol = AllChem.AssignBondOrdersFromTemplate(template, rd_mol)
    except Exception:
        Chem.MolToSmiles(rd_mol)
        m_order = eval(rd_mol.GetProp("_smilesAtomOutputOrder"))

        rd_mol2 = Chem.RenumberAtoms(rd_mol, m_order)

        # print(Chem.MolToSmiles(template))
        test_rd_mol = Chem.MolFromSmiles(Chem.MolToSmiles(rd_mol))
        Chem.MolToSmiles(test_rd_mol)
        m1_order = eval(test_rd_mol.GetProp("_smilesAtomOutputOrder"))
        test_rd_mol = Chem.RenumberAtoms(test_rd_mol, m1_order)

        mols = []
        for atom in rd_mol2.GetAtoms():
            mols.append(atom)
        conv = False
        for i, atom in enumerate(test_rd_mol.GetAtoms()):
            if mols[i].GetSymbol() != atom.GetSymbol():
                conv = True
        if conv:
            raise Exception("Ligand ordering error")
        else:
            return rd_mol
    return new_mol


def get_ligand_block(pdb_file, res_name, res_chain, res_num):
    """Get the block of a ligand from a PDB file."""
    block = ""
    tmp_block = ""
    parsed_atoms = []
    with open(pdb_file) as f:
        for line in f:
            # line = line.replace(res_name, ' '+res_name+ ' ')
            tmp_line = line
            if "HETATM" in tmp_line:
                if tmp_line.replace("HETATM", "")[0] != " ":
                    tmp_line = tmp_line.replace("HETATM", "HETATM ")
                if res_name in tmp_line:
                    if tmp_line.split(res_name)[0][-1] != " ":
                        tmp_line = tmp_line.replace(
                            tmp_line.split(res_name)[0][-1] + res_name, " " + res_name
                        )
                if (
                    str(res_num) in tmp_line
                    and not tmp_line.split(str(res_num))[0][-1].isnumeric()
                ):
                    if (
                        tmp_line.split(str(res_num))[0][-1] != " "
                        and str(res_num) not in tmp_line.split()[2]
                    ):
                        tmp_line = tmp_line.replace(str(res_num), " " + str(res_num))
                if len(tmp_line.split()) > 5:
                    if (
                        tmp_line.split()[0] == "HETATM"
                        and tmp_line.split()[3] == res_name
                        and tmp_line.split()[5] == str(res_num)
                        and tmp_line.split()[4] == res_chain
                    ):
                        # and line.split()[2][0] != 'H'\
                        parsed = (
                            tmp_line.split()[2],
                            tmp_line.split()[3],
                            tmp_line.split()[4],
                            tmp_line.split()[5],
                        ) in parsed_atoms
                        if not parsed:
                            parsed_atoms.append(
                                (
                                    tmp_line.split()[2],
                                    tmp_line.split()[3],
                                    tmp_line.split()[4],
                                    tmp_line.split()[5],
                                )
                            )
                            block += line
                            tmp_block += tmp_line
    return block, tmp_block


def _get_ligands(
    pdb_id,
    p,
    file_path,
):
    """Get the ligands from a PDB file."""
    import warnings

    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    warnings.filterwarnings("ignore")

    # unzip file
    if len(p.df["HETATM"]) > 0:
        pdb_tmp_path = file_path.split(".")[0] + ".pdb"
        with gzip.open(file_path, "rb") as f_in:
            with open(pdb_tmp_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        parser = PDBParser()
        structure = parser.get_structure(pdb_id.upper(), pdb_tmp_path)

        first_atom_idx = p.df["HETATM"]["atom_number"][0]

        main_df = p.amino3to1()
        main_chains = []
        main_ligands = []
        for chain in main_df["chain_id"].unique():
            main_chain_length = len(main_df[main_df.chain_id == chain])
            main_chain = list(structure[0][chain].get_residues())[:main_chain_length]
            main_ligand = list(structure[0][chain].get_residues())[main_chain_length:]
            # remove water
            main_ligand = [
                x
                for x in main_ligand
                if x.get_resname() != "HOH"
                and x.get_resname() != "DOD"
                and x.get_resname() != "UNK"
                and x.get_resname() != "UNL"
                and x.get_resname() != "UNX"
                and x.get_resname() not in D3TO1.keys()
                and len("".join([i for i in x.get_resname() if not i.isdigit()])) > 2
            ]

            # if len(main_ligand) >= 30:
            main_chains += main_chain
            main_ligands.append(main_ligand)

        other_chains = []
        for chain in structure[0].get_chains():
            if chain.id not in main_df["chain_id"].unique():
                other_chains.append(list(chain.get_residues()))

        all_connection = parse_pdb_file(pdb_tmp_path, minimum=first_atom_idx)

        # test covalent bonds between chains
        covbonds_ligand = []
        covbonds_cl = []

        for main_ligand in main_ligands:
            covbond_ligand, _, _ = get_covalent_bonds(
                main_ligand, main_ligand, all_connection, Time_profiling=False
            )
            covbonds_ligand.append(covbond_ligand)

            covbon_cl, A2M_C, A2M_L = get_covalent_bonds(
                main_chain, main_ligand, all_connection, Time_profiling=False
            )
            covbonds_cl.append(covbon_cl)

        covbonds = []
        interbonds = []

        for chain in other_chains:
            c1, _, c3 = get_covalent_bonds(chain, chain, all_connection)
            interbonds.append(c1)
            # A2M_Ls.append(c3)
            # print("covalent bonds between main chain and chain", chain[0].get_parent().id)
            c1, _, _ = get_covalent_bonds(main_chain, chain, all_connection)
            covbonds.append(c1)

        components = sum(main_ligands, []) + sum(other_chains, [])

        atom2component = np.zeros(len(list(structure[0].get_atoms())))
        for i, component in enumerate(main_chain):
            for atom in component.get_atoms():
                try:
                    atom2component[atom.get_serial_number()] = 0
                except Exception:
                    pass
        for i, component in enumerate(components):
            for atom in component.get_atoms():
                try:
                    atom2component[atom.get_serial_number()] = i + 1
                except Exception:
                    pass

        connectivity = np.zeros((len(components) + 1, len(components) + 1))
        id2component = {}
        connectivity[-1, -1] = 1
        for i, component in enumerate(components):
            id2component[component.get_id()[1]] = i
            connectivity[i, i] = 1

        for covbond_ligand in covbonds_ligand:
            for conn in covbond_ligand:
                atom1, atom2 = conn
                if atom1 < len(atom2component) and atom2 < len(atom2component):
                    connectivity[int(atom2component[atom1])][
                        int(atom2component[atom2])
                    ] = 1
                    connectivity[int(atom2component[atom2])][
                        int(atom2component[atom1])
                    ] = 1

        for covbon_cl in covbonds_cl:
            for conn in covbon_cl:
                atom1, atom2 = conn
                if atom1 < len(atom2component) and atom2 < len(atom2component):
                    connectivity[int(atom2component[atom1])][
                        int(atom2component[atom2])
                    ] = 1
                    connectivity[int(atom2component[atom2])][
                        int(atom2component[atom1])
                    ] = 1

        for i, covbon in enumerate(covbonds):
            for conn in covbon:
                atom1, atom2 = conn
                if atom1 < len(atom2component) and atom2 < len(atom2component):
                    connectivity[int(atom2component[atom1])][
                        int(atom2component[atom2])
                    ] = 1
                    connectivity[int(atom2component[atom2])][
                        int(atom2component[atom1])
                    ] = 1

        for i, covbon in enumerate(interbonds):
            for conn in covbon:
                atom1, atom2 = conn
                if atom1 < len(atom2component) and atom2 < len(atom2component):
                    connectivity[int(atom2component[atom1])][
                        int(atom2component[atom2])
                    ] = 1
                    connectivity[int(atom2component[atom2])][
                        int(atom2component[atom1])
                    ] = 1
        # connectivity
        idependent_components = []
        for i in range(len(components) + 1):
            idependent_components.append([i])

        idependent_components = merge_components(idependent_components, connectivity)

        chain2ligands = {}
        # Atom_seq = []
        # X = []
        # lengths = []
        chain2molecule = {}
        for id, idependent_component in enumerate(idependent_components):
            atomic_sequence = []
            coordinates = []
            length = 0
            for i in idependent_component:
                if i > 0:
                    try:
                        chain2molecule[components[i - 1].get_parent().id].append(
                            components[i - 1].get_resname()
                        )
                    except Exception:
                        chain2molecule[components[i - 1].get_parent().id] = [
                            components[i - 1].get_resname()
                        ]
                    if id > 0:
                        """for atom in components[i-1].get_atoms():
                            atomic_sequence.append(atom.get_name().replace("'", ''))
                            coordinates.append(atom.get_coord())
                            length += 1

                        res_name = components[i-1].get_resname()
                        chem_desc = describe_chemical(f"{res_name}")
                        sub_smiles = chem_desc["rcsb_chem_comp_descriptor"]["smiles"]"""

                        res_name = components[i - 1].get_resname()
                        chain_name = components[i - 1].get_full_id()[2]
                        res_number = components[i - 1].get_full_id()[3][1]
                        lig_block, tmp_lig_block = get_ligand_block(
                            pdb_tmp_path, res_name, chain_name, res_number
                        )
                        ligand_mol = process_ligand(lig_block, res_name, tmp_lig_block)

                        sub_smiles = Chem.MolToSmiles(ligand_mol)
                        smiles_order = ligand_mol.GetProp("_smilesAtomOutputOrder")
                        # print(other_mol.GetProp('_smilesAtomOutputOrder'))

                        pdb_atom_list = []
                        pdb_atom_list_names = []
                        for atom in components[i - 1].get_unpacked_list():
                            if (
                                atom.get_name()[0] != "H"
                                and atom.get_name() not in pdb_atom_list_names
                            ):
                                pdb_atom_list.append(atom)
                                pdb_atom_list_names.append(atom.get_name())

                        for j in range(len(pdb_atom_list)):
                            try:
                                atom = pdb_atom_list[eval(smiles_order)[j]]
                            except Exception:
                                print(
                                    "eval error",
                                    pdb_tmp_path,
                                    len(eval(smiles_order)),
                                    len(pdb_atom_list),
                                )
                                atom = pdb_atom_list[eval(smiles_order)[j]]
                            atomic_sequence.append(atom.get_name().replace("'", ""))
                            coordinates.append(atom.get_coord())
                            length += 1
            if id > 0:
                # print("Atomic sequence", np.array(atomic_sequence))
                # print("Coordinates", np.array(coordinates)[:10])
                try:
                    chain2ligands[components[i - 1].get_parent().id].append(
                        {
                            "seq": "-".join(
                                chain2molecule[components[i - 1].get_parent().id]
                            ),
                            "smiles": sub_smiles,
                            "atoms": atomic_sequence,
                            "X": np.array(coordinates),
                            "length": length,
                            "chain": components[i - 1].get_parent().id,
                        }
                    )
                except Exception:
                    chain2ligands[components[i - 1].get_parent().id] = [
                        {
                            "seq": "-".join(
                                chain2molecule[components[i - 1].get_parent().id]
                            ),
                            "smiles": sub_smiles,
                            "atoms": atomic_sequence,
                            "X": np.array(coordinates),
                            "length": length,
                            "chain": components[i - 1].get_parent().id,
                        }
                    ]
                # Atom_seq.append(''.join(atomic_sequence))
                # X.append(coordinates)
                # lengths.append(length)

        os.remove(pdb_tmp_path)
        return chain2ligands
    else:
        return None


def _load_smiles(dir):
    """Load biounits and group their sequences by PDB and similarity (90%)."""
    smiles_dict = defaultdict(lambda: [])

    for file in tqdm([x for x in os.listdir(dir) if x.endswith(".pickle")]):
        load_path = os.path.join(dir, file)
        if os.path.isdir(load_path):
            continue
        with open(load_path, "rb") as f:
            pdb_dict = pickle.load(f)
        smiles = [
            (
                chain,
                "".join(
                    np.array(
                        list(".".join(lg["smiles"] for lg in pdb_dict[chain]["ligand"]))
                    ).tolist()
                ),
            )
            for chain in pdb_dict.keys()
            if "ligand" in pdb_dict[chain].keys()
        ]
        smiles = [(chain, sm) for chain, sm in smiles if len(sm) > 0]
        smiles_dict[file[:4]] += smiles

    return smiles_dict


def _unique_chains(seqs_list):
    """Get unique chains."""
    new_seqs_list = [seqs_list[0]]
    chains = [new_seqs_list[0][0]]

    for seq in seqs_list[1:]:
        if seq[0] not in chains:
            new_seqs_list.append(seq)
            chains.append(seq[0])

    return new_seqs_list


def _merge_chains_ligands(seqs_dict_):
    """Look into the chains of each PDB and regroup redundancies (at 90% sequence identity)."""
    seqs_dict = copy.deepcopy(seqs_dict_)
    pdbs_to_delete = []

    for pdb in tqdm(seqs_dict.keys()):
        if seqs_dict[pdb] == []:
            pdbs_to_delete.append(pdb)
            continue

        seqs_dict[pdb] = _unique_chains(seqs_dict[pdb])
        groups, ref_sms, indexes = [], [], []

        for k in range(len(seqs_dict[pdb])):
            if k in indexes:
                continue
            group = [seqs_dict[pdb][k][0]]
            ref_sm = seqs_dict[pdb][k][1]
            ref_sms.append(ref_sm)
            indexes.append(k)

            for i in range(k + 1, len(seqs_dict[pdb])):
                chain, seq = seqs_dict[pdb][i][0], seqs_dict[pdb][i][1]
                if (
                    i in indexes
                    or len(seq) > 1.1 * len(ref_sm)
                    or len(seq) < 0.9 * len(ref_sm)
                    or calculate_tanimoto_distance(seq, ref_sm) > 0.1
                ):
                    continue
                group.append(chain)
                indexes.append(i)

            groups.append(group)

        new_group = []
        for group, seq in zip(groups, ref_sms):
            new_group.append(("-".join(group), seq))
        seqs_dict[pdb] = new_group

    for pdb in pdbs_to_delete:
        del seqs_dict[pdb]

    return seqs_dict


def save_binary_vectors_to_file(vectors, file_path):
    """Save binary vectors to a file."""
    with open(file_path, "wb") as file:
        for vector in vectors:
            bytes_vector = np.packbits(vector)  # Convert binary array to bytes
            file.write(bytes_vector)


def get_binary_vector_by_index(file_path, index, vector_length):
    """Get a binary vector from a file by index."""
    # vector_length = 2048
    vector_byte_size = (vector_length + 7) // 8  # Number of bytes per vector

    with open(file_path, "rb") as file:
        with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            start = index * vector_byte_size
            end = start + vector_byte_size

            bytes_vector = mm[start:end]
            binary_vector = np.unpackbits(np.frombuffer(bytes_vector, dtype=np.uint8))

    return binary_vector[:vector_length]  # Trim the array to the original size


def calculate_fingerprint_similarity(fp1, fp2):
    """Calculate the Tanimoto similarity between two molecules."""
    # fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
    # fp1 = fpgen.GetFingerprint(mol1)
    # fp2 = fpgen.GetFingerprint(mol2)
    # bit_string = ''.join(str(bit) for bit in fp1)
    # bfp1 = DataStructs.CreateFromBitString(bit_string)

    # bit_string = ''.join(str(bit) for bit in fp2)
    # bfp2 = DataStructs.CreateFromBitString(bit_string)

    similarity = DataStructs.FingerprintSimilarity(fp1, fp2)
    return similarity


def calculate_tanimoto_similarity(mol1, mol2):
    """Calculate the Tanimoto similarity between two molecules."""
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
    fp1 = fpgen.GetFingerprint(mol1)
    fp2 = fpgen.GetFingerprint(mol2)
    similarity = DataStructs.FingerprintSimilarity(fp1, fp2)
    return similarity


def calculate_similarity(args, vector_length=2048, root_path="bin_fps"):
    """Calculate the Tanimoto distance between two fp vectors."""
    i, j = args
    # i, j, txt_file = args

    # sm1 = get_line_by_index(txt_file, i)
    # sm2 = get_line_by_index(txt_file, j)

    # molecule1 = Chem.MolFromSmiles(sm1)
    # molecule2 = Chem.MolFromSmiles(sm2)
    # fp1, fp2 = get_binary_vector_by_index(file_path, i, vector_length), get_binary_vector_by_index(file_path, j, vector_length)
    with open(root_path + str(i) + ".pickle", "rb") as file:
        bfp1 = pickle.load(file)

    with open(root_path + str(j) + ".pickle", "rb") as file:
        bfp2 = pickle.load(file)

    fp1 = ExplicitBitVect(2048)
    fp1.FromBase64(bfp1)
    fp2 = ExplicitBitVect(2048)
    fp2.FromBase64(bfp2)

    similarity = calculate_fingerprint_similarity(fp1, fp2)
    return i, j, 1 - similarity


def _compare_lig_identity(lig, ligs, threshold):
    """Assess whether a sequence is identical to another one in a list of sequences."""
    mol1 = Chem.MolFromSmiles(lig)
    for lig2 in ligs:
        mol2 = Chem.MolFromSmiles(lig2)
        if calculate_tanimoto_similarity(mol1, mol2) <= threshold:
            return True
    return False


def _compare_smiles(ligs1, ligs2, threshold):
    """Assess whether 2 lists of sequences contain exactly the same set of sequences."""
    for lig in ligs1:
        if not _compare_lig_identity(lig, ligs2, threshold):
            return False

    for lig in ligs2:
        if not _compare_lig_identity(lig, ligs1, threshold):
            return False

    return True


def calculate_tanimoto_distance(sm1, sm2):
    """Calculate the Tanimoto distance between two SMILES."""
    mol1 = Chem.MolFromSmiles(sm1)
    mol2 = Chem.MolFromSmiles(sm2)
    return 1 - calculate_tanimoto_similarity(mol1, mol2)


def get_line_by_index(file_path, index):
    """Get a line from a file by index."""
    with open(file_path) as file:
        with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            start = mm.find(b"\n", mm.find(b"\n", 0) + 1)  # Skip the first line

            for _ in range(index):
                start = mm.find(b"\n", start + 1)

            end = mm.find(b"\n", start + 1)
            line = mm[start + 1 : end].decode().strip()

    return line


# Define a function to perform Tanimoto clustering
def perform_tanimoto_clustering(smiles_list, threshold, tmp_folder):
    """Perform Tanimoto clustering."""
    from multiprocessing import Pool

    unique_smiles = []
    smiles_mapping = []
    print("Mapping unique smiles for Tanimoto...")
    counter = 0
    for i, sm in tqdm(enumerate(smiles_list)):
        if sm not in unique_smiles:
            unique_smiles.append(sm)
            smiles_mapping.append(counter)
            counter += 1
        else:
            smiles_mapping.append(unique_smiles.index(sm))
    # Convert SMILES to RDKit molecules
    # molecules = [Chem.MolFromSmiles(smiles) for smiles in unique_smiles]

    # Calculate pairwise similarities
    # similarity_matrix = np.zeros((len(molecules), len(molecules)))

    print("Calculating fingerprints...")
    fpSize = 2048
    tmp_root_path = "./tmp_fps/"
    # filename = tmp_root_path+"bin_fps.h5"

    if os.path.exists(tmp_root_path):
        # Remove the old directory and its contents
        for root, dirs, files in os.walk(tmp_root_path, topdown=False):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                os.remove(file_path)
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                os.rmdir(dir_path)
        os.rmdir(tmp_root_path)

    # Create the new directory
    os.makedirs(tmp_root_path)

    # FPs = np.zeros((len(unique_smiles), fpSize), dtype=int)
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=fpSize)
    # fp1 = fpgen.GetFingerprint(mol1)
    for i, sm in tqdm(enumerate(unique_smiles)):
        mol = Chem.MolFromSmiles(sm)
        fp = fpgen.GetFingerprint(mol)
        # data = base64.b64decode(fp.ToBase64())
        with open(tmp_root_path + str(i) + ".pickle", "wb") as file:
            pickle.dump(fp.ToBase64(), file)

    print("Preparing parallel Tanimoto...")
    # with open("unique_smiles.txt", "w") as file:
    #    for item in unique_smiles:
    #        file.write(item + "\n")

    similarity_matrix = []
    args_list = []
    for i in tqdm(range(len(unique_smiles))):
        for j in range(i):
            args_list.append((i, j))
            # args_list.append((i, j, "unique_smiles.txt"))

    print("Constructing distance matrix...")
    with Pool() as pool:
        # results = pool.map(calculate_similarity, args_list)
        results = []
        for result in tqdm(
            pool.imap_unordered(
                partial(
                    calculate_similarity, vector_length=fpSize, root_path=tmp_root_path
                ),
                args_list,
            ),
            total=len(args_list),
        ):
            results.append(result)

    square_dist = np.zeros((len(unique_smiles), len(unique_smiles)))
    for i, j, distance in sorted(results):
        square_dist[i][j] = distance

    print("Saving distance matrix...")
    np.save(tmp_folder + "/" + "Tanimoto_distance_matrix.npy", square_dist)
    N = len(smiles_list)
    results = None
    unique_smiles = None
    smiles_list = None
    args_list = None

    print("Copying results...")
    counter = 0
    for i in tqdm(range(N)):
        for j in range(i):
            # similarity_matrix.append(np.random.random())
            counter += 1
    similarity_matrix = np.zeros(counter)
    k = 0
    for i in tqdm(range(N)):
        for j in range(i):
            Ii = max(smiles_mapping[i], smiles_mapping[j])
            J = min(smiles_mapping[i], smiles_mapping[j])
            similarity_matrix[k] = square_dist[Ii][J]
            k += 1
    # Perform clustering
    # dist_matrix = 1 - similarity_matrix
    # print(similarity_matrix)
    print("Clustering with Butina")
    clusters = Butina.ClusterData(similarity_matrix, N, threshold, isDistData=True)

    if os.path.exists(tmp_root_path):
        # Remove the old directory and its contents
        for root, dirs, files in os.walk(tmp_root_path, topdown=False):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                os.remove(file_path)
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                os.rmdir(dir_path)
        os.rmdir(tmp_root_path)

    return clusters


def _run_tanimoto_clustering(smiles_dict, threshold, tmp_folder):
    """Run Tanimoto clustering on a dictionary of SMILES."""
    smiles_list = []
    chains_list = []
    for k in smiles_dict.keys():
        for c in smiles_dict[k]:
            chains_list.append(k + "_" + c[0])
            smiles_list.append(c[1])
    clusters = perform_tanimoto_clustering(smiles_list, threshold, tmp_folder)
    chain_cluster_dict = {}
    pdb_cluster_dict = {}
    for i, cluster in enumerate(clusters):
        chain_cluster_dict[chains_list[cluster[0]]] = list(
            [chains_list[cl] for cl in cluster]
        )
        pdb_cluster_dict[chains_list[cluster[0]]] = np.array(
            list(set(list([chains_list[cl].split("_")[0] for cl in cluster])))
        )
    return chain_cluster_dict, pdb_cluster_dict
