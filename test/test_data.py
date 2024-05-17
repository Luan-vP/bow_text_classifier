import pytest
from conftest import (
    parsed_test_data,
    test_data,
    test_tag_to_index,
    test_tag_to_index_check_unk,
    test_word_to_index,
    test_word_to_index_check_unk,
)


def test_parse_data(test_data, parsed_test_data):
    from bow_text_classifier.data import _parse_data

    output = _parse_data(test_data)

    assert output == parsed_test_data


def test_create_dict():
    from bow_text_classifier.data import _create_dict

    parsed_test_data = [
        ("4", "it 's a lovely film"),
        ("3", "no one goes unindicted here"),
    ]

    word_to_index_results = {
        "<unk>": 0,
        "it": 1,
        "'s": 2,
        "a": 3,
        "lovely": 4,
        "film": 5,
        "no": 6,
        "one": 7,
        "goes": 8,
        "unindicted": 9,
        "here": 10,
    }

    tag_to_index_results = {"4": 0, "3": 1}

    word_to_index, tag_to_index = _create_dict(parsed_test_data, check_unk=False)

    assert word_to_index == word_to_index_results
    assert tag_to_index == tag_to_index_results

    parsed_test_data_2 = [
        ("4", "it 's a cat friendly film"),
        ("1", "no one goes unindicted here"),
        ("5", "it 's a lovely film"),
    ]

    word_to_index_results_2 = {
        "<unk>": 0,
        "it": 1,
        "'s": 2,
        "a": 3,
        "lovely": 4,
        "film": 5,
        "no": 6,
        "one": 7,
        "goes": 8,
        "unindicted": 9,
        "here": 10,
        "cat": 0,
        "friendly": 0,
    }

    tag_to_index_results_2 = {"4": 0, "3": 1, "1": 2, "5": 3}

    word_to_index, tag_to_index = _create_dict(
        parsed_test_data_2, word_to_index, tag_to_index, check_unk=True
    )

    assert word_to_index == word_to_index_results_2
    assert tag_to_index == tag_to_index_results_2
