# -*- coding: utf-8 -*-

import os

from tqdm import tqdm
import requests
from typing import Union

REPO_BASE_URL = (
    os.environ.get("TRANSLITERATE_HIN2ENG_DATA_URL")
    or "https://github.com/in-rolls/indicate/raw/master/indicate/data/model/"
)


def download_file(url: str, target: typing.Union[str, bytes, os.PathLike]) -> None:

    if "TRANSLITERATE_AUTH_TOKEN" in os.environ:
        auth_token = os.environ["TRANSLITERATE_AUTH_TOKEN"]
        headers = {f"Authorization": "token {auth_token}"}
    else:
        headers = {}

    # Streaming, so we can iterate over the response.
    r = requests.get(url, stream=True, headers=headers)

    if r.status_code == 200:
        chunk_size = 64 * 1024
        # Total size in bytes.
        total_size = int(r.headers.get("content-length", 0)) / chunk_size

        total_size += 1

        with open(target, "wb") as f:
            for data in tqdm(r.iter_content(chunk_size), total=total_size, unit_scale=chunk_size / 1024, unit="KB"):
                f.write(data)
    else:
        raise Exception(f"Error: Status code is not 200. Status code {r.status_code}")

