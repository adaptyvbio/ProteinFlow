from aiobotocore.session import get_session
from botocore import UNSIGNED
from botocore.config import Config
import asyncio
import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool
import os
from p_tqdm import p_map


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
            except Exception as e:
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
    """mission for single process"""
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
