from setuptools import setup
from setuptools import find_packages


setup(
    name='bestprot',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        "numpy",
        "editdistance>=0.6.0",
        "Biopython>=1.80",
        "click>=8.1.3",
        "biopandas>=0.4.1",
        "boto3>=1.25.3",
        "p_tqdm>=1.4.0",
        "networkx==2.8.8",
        "einops>=0.6.0",
        "pandas>=1.5.2",
        "torch>=1.10.0",
        "biotite==0.35.0",
        "awscli",
        "rcsbsearch @ git+https://github.com/sbliven/rcsbsearch@dbdfe3880cc88b0ce57163987db613d579400c8e"
    ],
    author = "Adaptyv Biosystems",
    author_email = "contact@adaptyvbio.com",
    description = ("A library for running the BestProt protein data processing pipeline."),
    entry_points={
        "console_scripts": [
            "generate_bestprot = bestprot.scripts.generate_data:main",
            "download_bestprot = bestprot.scripts.download_data:main",
            "split_bestprot = bestprot.scripts.split_data:main"
        ]
    },
    python_requires=">=3.8,<3.10",
)
