"""Bump the version of the package in pyproject.toml and .conda/meta.yaml."""

import click


@click.option("--version", required=True)
@click.command()
def main(version):
    """
    Bump the version.

    Parameters
    ----------
    version : str
        The version to bump to
    """
    with open("pyproject.toml") as f:
        lines = f.readlines()
    with open("pyproject.toml", "w") as f:
        for line in lines:
            if line.startswith("version"):
                line = f'version = "{version}"\n'
            f.write(line)
    with open(".conda/meta.yaml") as f:
        lines = list(f.readlines())
    lines[0] = f'{{% set version = "{version}" %}}\n'
    with open(".conda/meta.yaml", "w") as f:
        for line in lines:
            f.write(line)


if __name__ == "__main__":
    main()
