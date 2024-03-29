# Contributing to ProteinFlow

Thank you for your interest in contributing to the ProteinFlow project! Your contributions are valuable and help make the project better. Please take a moment to review the guidelines for contributing.

## Table of Contents

- [Installation](#installation)
- [Testing](#testing)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Guidelines](#pull-request-guidelines)
- [Documentation](#documentation)

## Installation

To install ProteinFlow for development, please follow these steps:

1. Clone the repository:
    
    ```bash
    git clone https://github.com/adaptyvbio/ProteinFlow
    cd ProteinFlow
    ```
    
2. Create a Conda environment named `proteinflow` with Python 3.9:
    
    ```bash
    conda create --name proteinflow -y python=3.9
    ```
    
3. Activate the `proteinflow` environment:
    
    ```bash
    conda activate proteinflow
    ```
    
4. Install ProteinFlow in editable mode along with its dependencies and `pytest`:
    
    ```bash
    python -m pip install -e .
    python -m pip install "rcsbsearch @ git+https://github.com/sbliven/rcsbsearch@dbdfe3880cc88b0ce57163987db613d579400c8e"
    python -m pip install pytest
    conda install -c bioconda -c conda-forge mmseqs2
    ```
    
5. Install the pre-commit hooks:
    
    ```bash
    pre-commit install
    pre-commit install --hook-type commit-msg
    ```

Now you are ready to start developing with ProteinFlow!

## Testing

Run `pytest tests` to check that the code works correctly.

## Commit Guidelines

When making commits to the repository, please follow these guidelines:

- Commit messages should follow the format: `type(scope): description`.
- The `type` should be one of the following:
    - `feat`: A new feature
    - `fix`: A bug fix
    - `docs`: Documentation changes
    - `style`: Code style changes (e.g., formatting, missing semicolons)
    - `refactor`: Code refactorings that don't change functionality
    - `test`: Adding or modifying tests
    - `chore`: Other changes that don't modify code or tests
- The `scope` (optional) should indicate the module or feature area affected by the commit.
- The `description` should be a concise summary of the changes introduced by the commit.

The description should be uncapitalized, in imperative, present tense. The total length of the header shouldn't exceed 100 characters. If you want to add more information, it can be put in the body of the message, separated with a blank line.

Examples: 
- `'fix(processing): make ProteinDataset work with old style dictionaries'`,
- `'test: test for exclude_chains in test_generate'`.
- ```
  'feat(generating): add ligand support

  Add the option to load ligands, cluster using Tanimoto clustering on smiles, 
  and download files based on a pdb_id text file.'
  ```

See the [Conventional Commits guide](https://www.conventionalcommits.org/en/v1.0.0/) for more information.

Before committing changes, please ensure that you have installed the pre-commit hooks by running `pre-commit install` and `pre-commit install --hook-type commit-msg` in your local repository. This ensures that code formatting and linting checks pass before committing.

## Pull Request Guidelines

When creating a pull request, please adhere to the following guidelines:

- Use a capitalized and imperative present tense for the pull request title.
- Provide a clear and informative description of the changes introduced by the pull request.
- If the pull request addresses an issue, reference it in the description using the `#issue-number` syntax.

Your pull request will be reviewed by the project maintainers, and feedback or changes may be requested. Please be responsive to the feedback and make the necessary updates to ensure a smooth review process.

## Documentation

We use NumPy style docstrings to document our code. When contributing, please ensure that your code is documented using this style. NumPy style docstrings provide a consistent and readable format for documenting functions, classes, and modules.

For more information on how to write NumPy style docstrings, refer to the [NumPy Docstring Guide](https://numpydoc.readthedocs.io/en/latest/format.html).

Thank you for your contributions to ProteinFlow!
