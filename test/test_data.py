import pytest
from conftest import parsed_test_data, test_data


def test_parse_data(test_data, parsed_test_data):
    from bow_text_classifier.data import _parse_data

    output = _parse_data(test_data)

    assert output == parsed_test_data
