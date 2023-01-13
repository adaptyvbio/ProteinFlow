import click
from bestprot import generate_data


@click.option(
    "--tag",
    help="The name of the dataset",
)
@click.option(
    "--local_datasets_folder",
    default="./data",
    help="The folder where bestprot datasets, temporary files and logs will be stored"
)
@click.option(
    "--min_length",
    default=30,
    help="The minimum number of non-missing residues per chain",
)
@click.option(
    "--max_length",
    default=10000,
    help="The maximum number of residues per chain (set None for no threshold)",
)
@click.option("--resolution_thr", default=3.5, help="The maximum resolution")
@click.option(
    "--missing_ends_thr",
    default=0.3,
    help="The maximum fraction of missing residues at the ends",
)
@click.option(
    "--missing_middle_thr",
    default=0.1,
    help="The maximum fraction of missing residues in the middle (after missing ends are disregarded)",
)
@click.option(
    "--filter_methods",
    default=True,
    help="If `True`, only files obtained with X-ray or EM will be processed",
)
@click.option(
    "--remove_redundancies",
    default=False,
    help="If `True`, removes biounits that are doubles of others sequence wise",
)
@click.option(
    "--seq_identity_threshold",
    default=0.9,
    type=float,
    help="The threshold upon which sequences are considered as one and the same (default: 90%)",
)
@click.option(
    "--valid_split",
    default=0.05,
    type=float,
    help="The percentage of chains to put in the validation set (default 5%)",
)
@click.option(
    "--test_split",
    default=0.05,
    type=float,
    help="The percentage of chains to put in the test set (default 5%)",
)
@click.option(
    "--split_tolerance",
    default=0.2,
    type=float,
    help="The tolerance on the split ratio (default 20%)",
)
@click.option(
    "-n",
    default=None,
    type=int,
    help="The number of files to process (for debugging purposes)",
)
@click.option(
    "--force", is_flag=True, help="When `True`, rewrite the files if they already exist"
)
@click.option(
    "--pdb_snapshot",
    help="The pdb snapshot folder to load",
)
@click.command()
def main(**kwargs):
    generate_data(**kwargs)


if __name__ == "__main__":
    main()
