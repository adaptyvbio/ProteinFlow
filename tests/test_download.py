import subprocess
import shutil
from bestprot import ProteinLoader


def test_download():
    """Test download_data + split_data + ProteinLoader"""

    subprocess.run(["download_bestprot", "--tag", "test", "--skip_splitting"], check=True)
    subprocess.run(["split_bestprot", "--tag", "test"], check=True)
    valid_loader = ProteinLoader("./data/bestprot_test/valid", batch_size=8, node_features_type="chemical+sidechain_orientation+dihedral+secondary_structure", rewrite=True)
    batch = next(iter(valid_loader))
    assert set(batch.keys()) == {"X", "S", "mask", "mask_original", "residue_idx", "chain_encoding_all", "chain_id", "sidechain_orientation", "dihedral", "chemical", "secondary_structure"}
    assert batch["X"].shape == (8, batch["X"].shape[1], 4, 3)
    assert batch["S"].shape == (8, batch["X"].shape[1])
    assert batch["dihedral"].shape == (8, batch["X"].shape[1], 2)

    shutil.rmtree("./data/bestprot_test")

# test_download()