"""Test ProteinEntry class."""
import numpy as np

from proteinflow.data import ProteinEntry


def test_entry():
    """Test `ProteinEntry.from_pdb` and `ProteinEntry.to_pdb`."""
    entry = ProteinEntry.from_pdb(
        "./sample_data/7zor.pdb", heavy_chain="H", light_chain="L", antigen_chains=None
    )
    crd1 = entry.get_coordinates()
    seq1 = entry.get_sequence(encode=True)
    entry.to_pdb("./sample_data/7zor_copy.pdb", only_backbone=False)
    entry = ProteinEntry.from_pdb(
        "./sample_data/7zor_copy.pdb",
        heavy_chain="H",
        light_chain="L",
        antigen_chains=None,
    )
    crd2 = entry.get_coordinates()
    seq2 = entry.get_sequence(encode=True)
    assert np.allclose(crd1, crd2)
    assert np.all(seq1 == seq2)


if __name__ == "__main__":
    test_entry()
