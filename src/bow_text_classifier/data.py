"""
This module contains functions for downloading and parsing datasets.
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


def create_training_datasets():
    for dataset_name, url in DATASETS.items():
        download_dataset(url, get_dataset_filepath(dataset_name))

    with open(get_dataset_filepath("train"), "r") as f:
        train_data = _parse_data(f.read())

    with open(get_dataset_filepath("test"), "r") as f:
        test_data = _parse_data(f.read())

    word_to_index, tag_to_index = _create_dict(train_data)

    word_to_index, tag_to_index = _create_dict(
        test_data, word_to_index, tag_to_index, check_unk=True
    )

    train_data = list(_create_tensors(train_data, word_to_index, tag_to_index))
    test_data = list(_create_tensors(test_data, word_to_index, tag_to_index))

    return train_data, test_data, word_to_index, tag_to_index


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


def _parse_data(input: str) -> list[tuple[str, str]]:

    output = []

    for line in input.lower().strip().split("\n"):
        label, text = line.split(" ||| ")
        output.append((label, text))

    return output


# create word to index dictionary and tag to index dictionary from data
def _create_dict(
    data: list[tuple[str, str]],
    word_to_index: dict[str, int] = {},
    tag_to_index: dict[str, int] = {},
    check_unk=False,
):
    """
    Create word_to_index dictionary and tag_to_index dictionary from data.

    Args:
        data (list[tuple[str, str]]): List of tuples containing tag and sentences.
        word_to_index (dict[str, int], optional): Dictionary mapping words to indices. Defaults to {}.
        tag_to_index (dict[str, int], optional): Dictionary mapping tags to indices. Defaults to {}.
        check_unk (bool, optional): Whether to check for unknown words. Defaults to False.
    """
    # TODO - Consider decoupling the input dictionary with deepcopy()

    if not word_to_index:
        word_to_index["<unk>"] = 0

    for tag, sentence in data:
        for word in sentence.split(" "):
            if check_unk:
                if word not in word_to_index:
                    word_to_index[word] = word_to_index["<unk>"]
            else:
                if word not in word_to_index:
                    word_to_index[word] = len(word_to_index)

        if tag not in tag_to_index:
            tag_to_index[tag] = len(tag_to_index)

    return word_to_index, tag_to_index


def _create_tensors(data, word_to_index, tag_to_index):
    for tag, sentence in data:
        # N.B. The order changes here to ([sentence], tag)
        yield (
            [word_to_index[word] for word in sentence.split(" ")],
            tag_to_index[tag],
        )
