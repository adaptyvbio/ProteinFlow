@click.option(
    "--tmp_folder",
    default="./data/tmp_pdb",
    help="The folder where temporary files will be saved",
)
@click.option(
    "--output_folder",
    default="./data/pdb",
    help="The folder where the output files will be saved",
)
@click.option(
    "--log_folder",
    default="./data/logs",
    help="The folder where the log file will be saved",
)
@click.option(
    "--out_split_dict_folder",
    default="./data/dataset_splits_dicts",
    help="The folder where the dictionaries containing the train/validation/test splits information will be saved",
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
    "--split_database", default=False, help="Whether or not to split the database"
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
@click.command()
def main(
    tmp_folder="./data/tmp_pdb",
    output_folder="./data/pdb",
    log_folder="./data/logs",
    min_length=30,
    max_length=10000,
    resolution_thr=3.5,
    missing_ends_thr=0.3,
    missing_middle_thr=0.1,
    filter_methods=True,
    remove_redundancies=False,
    seq_identity_threshold=0.9,
    n=None,
    force=False,
    split_tolerance=0.2,
    split_database=False,
    test_split=0.05,
    valid_split=0.05,
    out_split_dict_folder="./data/dataset_splits_dicts",
):
    _run_processing(
        tmp_folder,
        output_folder,
        log_folder,
        min_length,
        max_length,
        resolution_thr,
        missing_ends_thr,
        missing_middle_thr,
        filter_methods,
        remove_redundancies,
        seq_identity_threshold,
        n,
        force,
        split_tolerance,
        split_database,
        test_split,
        valid_split,
        out_split_dict_folder,
    )


if __name__ == "__main__":
    main()
