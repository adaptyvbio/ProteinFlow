import asyncio
import os
import shutil
import subprocess
from operator import attrgetter

from aiobotocore.session import get_session
from botocore import UNSIGNED
from botocore.config import Config
from p_tqdm import p_map

from proteinflow.constants import S3Obj


def _download_dataset_dicts_from_s3(dict_folder_path, s3_path):
    """
    Download dictionaries containing database split information from s3 to a local folder
    """

    train_path = os.path.join(s3_path, "train.pickle")
    valid_path = os.path.join(s3_path, "valid.pickle")
    test_path = os.path.join(s3_path, "test.pickle")
    classes_path = os.path.join(s3_path, "classes.pickle")

    if not os.path.exists(dict_folder_path):
        os.makedirs(dict_folder_path)

    subprocess.run(
        ["aws", "s3", "cp", "--no-sign-request", train_path, dict_folder_path]
    )
    subprocess.run(
        ["aws", "s3", "cp", "--no-sign-request", valid_path, dict_folder_path]
    )
    subprocess.run(
        ["aws", "s3", "cp", "--no-sign-request", test_path, dict_folder_path]
    )
    subprocess.run(
        ["aws", "s3", "cp", "--no-sign-request", classes_path, dict_folder_path]
    )


def _s3list(
    bucket,
    path,
    start=None,
    end=None,
    recursive=True,
    list_dirs=True,
    list_objs=True,
    limit=None,
):
    """Iterator that lists a bucket's objects under path, (optionally) starting with start and ending before end.

    If recursive is False, then list only the "depth=0" items (dirs and objects).

    If recursive is True, then list recursively all objects (no dirs).

    Parameters
    ----------
    bucket:
        a boto3.resource('s3').Bucket().
    path:
        a directory in the bucket.
    start:
        optional: start key, inclusive (may be a relative path under path, or
        absolute in the bucket)
    end:
        optional: stop key, exclusive (may be a relative path under path, or
        absolute in the bucket)
    recursive:
        optional, default True. If True, lists only objects. If False, lists
        only depth 0 "directories" and objects.
    list_dirs:
        optional, default True. Has no effect in recursive listing. On
        non-recursive listing, if False, then directories are omitted.
    list_objs:
        optional, default True. If False, then directories are omitted.
    limit:
        optional. If specified, then lists at most this many items.

    Returns
    -------
    iterator
        an iterator of S3Obj.
    """

    kwargs = dict()
    if start is not None:
        if not start.startswith(path):
            start = os.path.join(path, start)
    if end is not None:
        if not end.startswith(path):
            end = os.path.join(path, end)
    if not recursive:
        kwargs.update(Delimiter="/")
        if not path.endswith("/") and len(path) > 0:
            path += "/"
    kwargs.update(Prefix=path)
    if limit is not None:
        kwargs.update(PaginationConfig={"MaxItems": limit})

    paginator = bucket.meta.client.get_paginator("list_objects")
    for resp in paginator.paginate(Bucket=bucket.name, **kwargs):
        q = []
        if "CommonPrefixes" in resp and list_dirs:
            q = [S3Obj(f["Prefix"], None, None, None) for f in resp["CommonPrefixes"]]
        if "Contents" in resp and list_objs:
            q += [
                S3Obj(f["Key"], f["LastModified"], f["Size"], f["ETag"])
                for f in resp["Contents"]
            ]
        # note: even with sorted lists, it is faster to sort(a+b)
        # than heapq.merge(a, b) at least up to 10K elements in each list
        q = sorted(q, key=attrgetter("key"))
        if limit is not None:
            q = q[:limit]
            limit -= len(q)
        for p in q:
            if end is not None and p.key >= end:
                return
            yield p


def _download_dataset_from_s3(
    dataset_path="./data/proteinflow_20221110/",
    s3_path="s3://ml4-main-storage/proteinflow_20221110/",
):
    """Download the pre-processed files."""
    if s3_path.startswith("s3"):
        print("Downloading the dataset from s3...")
        subprocess.run(
            ["aws", "s3", "sync", "--no-sign-request", s3_path, dataset_path]
        )
        print("Done!")
    else:
        shutil.move(s3_path, dataset_path)


def _get_s3_paths_from_tag(tag):
    """Get the path to the data and split dictionary folders on S3 given a tag."""
    dict_path = f"s3://proteinflow-datasets/{tag}/proteinflow_{tag}_splits_dict/"
    data_path = f"s3://proteinflow-datasets/{tag}/proteinflow_{tag}/"
    return data_path, dict_path


async def _getobj(client, key):
    resp = await client.get_object(Bucket="pdbsnapshots", Key=key)
    return await resp["Body"].read()


async def _download_file(client, snapshots, tmp_folder, id):
    pdb_id, biounit = id.lower().split("-")
    prefixes = [
        "/pub/pdb/data/biounit/PDB/all/",
        "/pub/pdb/data/biounit/mmCIF/all/",
        "/pub/pdb/data/assemblies/mmCIF/all/",
    ]
    types = ["pdb", "cif", "cif"]
    filenames = {
        "cif": f"{pdb_id}-assembly{biounit}.cif.gz",
        "pdb": f"{pdb_id}.pdb{biounit}.gz",
    }
    for folder in snapshots:
        for prefix, t in zip(prefixes, types):
            file = folder + prefix + filenames[t]
            local_path = os.path.join(tmp_folder, f"{pdb_id}-{biounit}") + f".{t}.gz"
            try:
                obj = await _getobj(client, file)
                with open(local_path, "wb") as f:
                    f.write(obj)
                return local_path
            except Exception:
                pass
    return id


async def _go(download_list, tmp_folder, snapshots):
    session = get_session()
    async with session.create_client(
        "s3", config=Config(signature_version=UNSIGNED)
    ) as client:
        tasks = [
            _download_file(
                client=client,
                snapshots=snapshots,
                # key=os.path.join(prefix, f"{x.split('-')[0]}.pdb{x.split('-')[1]}.gz"),
                tmp_folder=tmp_folder,
                id=x,
            )
            for x in download_list
        ]
        out = await asyncio.gather(*tasks)
    return out


def _singleProcess(download_list, tmp_folder, snapshots):
    """Mission for single process."""
    loop = asyncio.new_event_loop()
    return loop.run_until_complete(
        _go(download_list, tmp_folder=tmp_folder, snapshots=snapshots)
    )


def _download_s3_parallel(pdb_ids, tmp_folder, snapshots):
    # number of process
    no_tasks = max(16, len(pdb_ids) // 5000)

    download_list_chunk = [pdb_ids[i::no_tasks] for i in range(no_tasks)]
    out = p_map(
        lambda x: _singleProcess(
            download_list=x, tmp_folder=tmp_folder, snapshots=snapshots
        ),
        download_list_chunk,
    )
    return out
