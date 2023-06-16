import os
import shutil
from time import time

from proteinflow import ProteinLoader, generate_data


# @pytest.mark.skip()
def test_generate_sabdab():
    """Test generate_data with `sabdab=True`"""

    folder = "./data/proteinflow_test"
    if os.path.exists(folder):
        shutil.rmtree(folder)
    start = time()
    # generate_data(tag="test", n=50, sabdab=True, resolution_thr=1)
    generate_data(
        tag="test",
        sabdab=True,
        sabdab_data_path="./sample_data/sabdab.zip",
        require_antigen=True,
        skip_splitting=True,
    )
    end = time()
    assert all(["nan_nan" not in file for file in os.listdir(folder)])
    train_loader = ProteinLoader.from_args(
        dataset_folder=folder,
        batch_size=8,
        rewrite=True,
        max_length=1000,
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
        "cdr_id",  #
        "chain_dict",  #
        "cdr",  #
    }
    assert batch["X"].shape == (8, batch["X"].shape[1], 4, 3)
    assert batch["S"].shape == (8, batch["X"].shape[1])
    assert batch["masked_res"].shape == (8, batch["X"].shape[1])
    assert batch["X"].shape[1] <= 1000
    train_loader.dataset.set_cdr("H3")
    batch = next(iter(train_loader))
    assert len(batch["cdr_id"].unique()) == 1
    print(f"generation time: {end - start} sec")
    shutil.rmtree(folder)
    # start = time()
    # generate_data(tag="test", skip_splitting=True, sabdab=True, zip_path="./sample_data/sabdab.zip", require_antigen=True)
    # end = time()
    # print(f"generation time: {end - start} sec")
    # assert all(["nan_nan" not in file for file in os.listdir(folder)])
    # shutil.rmtree(folder)


if __name__ == "__main__":
    test_generate_sabdab()
