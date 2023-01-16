import click
from bestprot import download_data


@click.option(
    "--tag",
    help="The name of the dataset",
)
@click.option(
    "--local_datasets_folder",
    default="./data",
    help="The folder where bestprot datasets, temporary files and logs will be stored"
)
@click.command()
def main(**kwargs):
    download_data(**kwargs)


if __name__ == "__main__":
    main()
