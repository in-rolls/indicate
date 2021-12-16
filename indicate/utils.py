# -*- coding: utf-8 -*-

import os

from tqdm import tqdm
import requests


REPO_BASE_URL = (
    os.environ.get("TRANSLITERATE_HIN2ENG_DATA_URL")
    or "https://github.com/in-rolls/indicate/raw/master/indicate/data/model/"
)


def download_file(url, target):

    if "TRANSLITERATE_AUTH_TOKEN" in os.environ:
        auth_token = os.environ["TRANSLITERATE_AUTH_TOKEN"]
        headers = {"Authorization": "token {0!s}".format(auth_token)}
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
        return True
    else:
        print("ERROR: status_code={0:d}".format(r.status_code))
        return False
