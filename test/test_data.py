import pytest
from conftest import (
    create_tensors_results,
    long_parsed_test_data,
    parsed_test_data_2,
    simple_parsed_test_data,
    tag_to_index_results,
    tag_to_index_results_2,
    test_data,
    word_to_index_results,
    word_to_index_results_2,
)


def test_parse_data():
    from bow_text_classifier.data import _parse_data

    output = _parse_data(test_data)

    assert output == long_parsed_test_data


def test_create_dict():
    from bow_text_classifier.data import _create_dict

    word_to_index, tag_to_index = _create_dict(simple_parsed_test_data, check_unk=False)

    assert word_to_index == word_to_index_results
    assert tag_to_index == tag_to_index_results

    # Check that new words are tagged as unknown
    word_to_index, tag_to_index = _create_dict(
        parsed_test_data_2, word_to_index, tag_to_index, check_unk=True
    )

    assert word_to_index == word_to_index_results_2
    assert tag_to_index == tag_to_index_results_2


def test_create_tensors():
    from bow_text_classifier.data import _create_tensors

    tensors = list(
        _create_tensors(
            parsed_test_data_2, word_to_index_results_2, tag_to_index_results_2
        )
    )

    assert tensors == create_tensors_results
