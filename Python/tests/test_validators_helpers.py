from typing import Any

from pytest import mark

from random_forestry.validators import (
    negative_float,
    negative_integer,
    positive_float,
    positive_integer,
)


@mark.parametrize("test_input,expected", [(-1, False), (3, True), (1.4, False), (0, False)])
def test_positive_integer(test_input: Any, expected: bool):
    assert positive_integer(test_input) == expected


@mark.parametrize("test_input,expected", [(-1, False), (3, False), (1.4, True), (-2.7, False), (0, False)])
def test_positive_float(test_input: Any, expected: bool):
    assert positive_float(test_input) == expected


@mark.parametrize("test_input,expected", [(-1, True), (3, False), (-1.4, False), (0, False)])
def test_negative_integer(test_input: Any, expected: bool):
    assert negative_integer(test_input) == expected


@mark.parametrize("test_input,expected", [(-1, False), (3, False), (1.4, False), (0, False), (-3.4, True)])
def test_negative_float(test_input: Any, expected: bool):
    assert negative_float(test_input) == expected
