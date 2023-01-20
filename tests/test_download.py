import subprocess
import shutil


def test_download():
    """Test download_data."""
    subprocess.run(["download_bestprot", "--tag", "test"], check=True)
    shutil.rmtree("./data/bestprot_test")