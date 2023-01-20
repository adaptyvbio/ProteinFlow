import click
from bestprot import download_data


@click.option(
    "--tag",
    default="test",
    help="The name of the dataset",
)
@click.option(
    "--local_datasets_folder",
    default="./data",
    help="The folder where bestprot datasets, temporary files and logs will be stored"
)
@click.option(
    "--skip_splitting",
    is_flag=True,
    help="Use this flag to skip splitting the data"
)
@click.command(help="Download an existing BestProt dataset")
def main(**kwargs):
    download_data(**kwargs)


if __name__ == "__main__":
    main()
