"""Test generate_data with `sabdab=True`."""
import os
import shutil
from time import time

from proteinflow import generate_data
from proteinflow.data.torch import ProteinLoader


# @pytest.mark.skip()
def test_generate_brenda():
    """Test generate_data with `load_ligands=True`."""
    folder = "./data/proteinflow_brenda_sample/"
    if os.path.exists(folder):
        shutil.rmtree(folder)
    start = time()
    # generate_data(tag="test", n=50, sabdab=True, resolution_thr=1)
    generate_data(
        pdb_id_list_path="./sample_data/brenda_sample.csv",
        skip_splitting=True,
        load_ligands=True,
        require_ligand=True,
        n=50,
    )
    end = time()
    train_loader = ProteinLoader.from_args(
        dataset_folder=folder,
        batch_size=8,
        rewrite=True,
        max_length=1000,
        load_ligands=True,
    )
    batch = next(iter(train_loader))
    assert set(batch.keys()) == {
        "X",  #
        "S",  #
        "mask",  #
        "mask_original",  #
        "residue_idx",  #
        "chain_encoding_all",  #
        "chain_id",  #
        "masked_res",  #
        "pdb_id",  #
        "chain_dict",  #
        "ligand_smiles",  #
        "X_ligands",  #
        "ligand_lengths",  #
        "ligand_chains",  #
    }
    assert batch["X"].shape == (8, batch["X"].shape[1], 4, 3)
    assert batch["S"].shape == (8, batch["X"].shape[1])
    assert batch["masked_res"].shape == (8, batch["X"].shape[1])
    assert batch["X"].shape[1] <= 1000

    assert batch["X_ligands"].shape == (8, batch["X_ligands"].shape[1], 3)
    assert batch["ligand_chains"].shape == (8, batch["X_ligands"].shape[1], 1)
    assert len(batch["ligand_smiles"]) == 8
    assert batch["ligand_lengths"].shape == (8,)

    print(f"generation time: {end - start} sec")
    shutil.rmtree(folder)
    # start = time()
    # generate_data(tag="test", skip_splitting=True, sabdab=True, zip_path="./sample_data/sabdab.zip", require_antigen=True)
    # end = time()
    # print(f"generation time: {end - start} sec")
    # assert all(["nan_nan" not in file for file in os.listdir(folder)])
    # shutil.rmtree(folder)


if __name__ == "__main__":
    test_generate_brenda()
