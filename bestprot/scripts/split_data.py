import click
from bestprot import split_data


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
    "--ignore_existing",
    is_flag=True,
    help="Unless this flag is used, bestprot will not overwrite existing split dictionaries for this tag and will load them instead"
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
@click.command(help="Split an existing BestProt dataset into training, validation and test subset according to MMseqs clustering and homomer/heteromer/single chain proportions")
def main(**kwargs):
    split_data(**kwargs)


if __name__ == "__main__":
    main()
