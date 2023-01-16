import os
from setuptools import setup
from setuptools import find_packages

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='bestprot',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        "sidechainnet>=0.7.6",
        "numpy>=1.23.4",
        "editdistance>=0.6.0",
        "Biopython>=1.79",
        "click>=8.1.3",
        "biopandas>=0.4.1",
        "boto3>=1.25.3",
        "p_tqdm>=1.4.0",
        "networkx==2.8.8",
        "einops>=0.6.0",
        "pandas>=1.5.2",
        "torch>=1.13.1",
        "rcsbsearch @ git+https://github.com/sbliven/rcsbsearch@dbdfe3880cc88b0ce57163987db613d579400c8e"
    ],
    author = "Adaptyv Biosystems",
    author_email = "contact@adaptyvbio.com",
    description = ("A library for running the BestProt protein data processing pipeline."),
    entry_points={
        "console_scripts": [
            "generate_bestprot = bestprot.scripts.generate_data:main",
            "download_bestprot = bestprot.scripts.download_data:main",
        ]
    },
    python_requires=">=3.8",
)

# setup(
#     name = "bestprot",
#     version = "0.0.1",
#     author = "Adaptyv Biosystems",
#     author_email = "contact@adaptyvbio.com",
#     description = ("A library for using the BestProt protein data processing pipeline."),
#     license = "BSD",
#     keywords = "protein dataset",
#     # url = "http://packages.python.org/an_example_pypi_project",
#     packages=find_packages(),
#     long_description=read('README.md'),
#     classifiers=[
#         "Development Status :: 3 - Alpha",
#         "Topic :: Utilities",
#         "License :: OSI Approved :: BSD License",
#     ],
# )