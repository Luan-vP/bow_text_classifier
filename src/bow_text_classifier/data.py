"""
This module contains functions for downloading and parsing datasets.

N.B. It contains a side-effect that downloads the dataset files to the data
directory on import to ensure its availability.
"""

from logging import Logger
from pathlib import Path

import requests

logger = Logger(__name__)

data_dir = Path(__file__).parent.parent.parent / "data"

DATASETS = {
    "dev": "https://raw.githubusercontent.com/neubig/nn4nlp-code/master/data/classes/dev.txt",
    "train": "https://raw.githubusercontent.com/neubig/nn4nlp-code/master/data/classes/train.txt",
    "test": "https://raw.githubusercontent.com/neubig/nn4nlp-code/master/data/classes/test.txt",
}


def download_dataset(url: str, output_file: Path | str):
    output_file = Path(output_file)

    if not output_file.exists():
        logger.info(f"Downloading {url} to {output_file}")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with requests.get(url) as response:
            response.raise_for_status()

            with output_file.open("wb") as f:
                f.write(response.content)
    else:
        logger.info(f"File already exists: {output_file}. Skipping download.")

    return output_file


def get_dataset_filepath(dataset_name: str) -> Path:
    return data_dir / "classes" / (dataset_name + ".txt")


for dataset_name, url in DATASETS.items():
    download_dataset(url, get_dataset_filepath(dataset_name))
