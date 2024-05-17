import pytest
from conftest import (
    create_tensors_results_3,
    parsed_test_data_1,
    parsed_test_data_2,
    parsed_test_data_3,
    tag_to_index_results_2,
    tag_to_index_results_3,
    test_data_1,
    word_to_index_results_2,
    word_to_index_results_3,
)


def test_parse_data():
    from bow_text_classifier.data import _parse_data

    output = _parse_data(test_data_1)

    assert output == parsed_test_data_1


def test_create_dict():
    from bow_text_classifier.data import _create_dict

    word_to_index, tag_to_index = _create_dict(parsed_test_data_2, check_unk=False)

    assert word_to_index == word_to_index_results_2
    assert tag_to_index == tag_to_index_results_2

    # Check that new words are tagged as unknown
    word_to_index, tag_to_index = _create_dict(
        parsed_test_data_3,
        word_to_index_results_2,
        tag_to_index_results_2,
        check_unk=True,
    )

    assert word_to_index == word_to_index_results_2
    assert tag_to_index == tag_to_index_results_2


def test_create_tensors():
    from bow_text_classifier.data import _create_tensors

    tensors = list(
        _create_tensors(
            parsed_test_data_3, word_to_index_results_2, tag_to_index_results_2
        )
    )

    assert tensors == create_tensors_results_3
