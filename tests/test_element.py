import numpy as np
import pytest

from kwave.utils.kwave_array import Element


def test_element_equality_type_error():
    element = Element(group_id=0, type="rect", dim=2, active=True, measure=1, position=[1, 1, 1])

    with pytest.raises(TypeError, match=r"not an element with <class 'str'> is not of type Element"):
        element == "not an element"


def test_element_equality_is_true():
    element1 = Element(group_id=0, type="rect", dim=2, active=True, measure=1, position=[1, 1, 1])
    element2 = Element(group_id=0, type="rect", dim=2, active=True, measure=1, position=[1, 1, 1])
    assert element1 == element2


def test_element_equality_is_false():
    element1 = Element(group_id=0, type="rect", dim=2, active=True, measure=1, position=[1, 1, 1])
    # position field differs in element 2
    element2 = Element(group_id=0, type="rect", dim=2, active=True, measure=1, position=[2, 2, 2])
    assert element1 != element2

    # measure field differs in element 3
    element3 = Element(group_id=0, type="rect", dim=2, active=True, measure=2, position=[1, 1, 1])
    assert element1 != element3


def test_element_is_approximately_equal():
    base_element = Element(group_id=0, type="rect", dim=2, active=True, measure=1, position=[1, 1, 1])

    # Test cases
    assert base_element.is_approximately_equal(Element(group_id=0, type="rect", dim=2, active=True, measure=1, position=[1, 1, 1]))
    assert base_element.is_approximately_equal(Element(group_id=0, type="rect", dim=2, active=True, measure=1, position=[1, 1, 1.000001]))
    assert not base_element.is_approximately_equal(Element(group_id=1, type="rect", dim=2, active=True, measure=2, position=[1, 1, 1]))
    assert not base_element.is_approximately_equal(Element(group_id=0, type="rect", dim=2, active=True, measure=1, position=[2, 2, 2]))
    assert not base_element.is_approximately_equal(
        Element(group_id=0, type="rect", dim=2, active=True, measure=1, position=[1, 1, float("nan")])
    )
    assert not base_element.is_approximately_equal(
        Element(group_id=0, type="rect", dim=2, active=True, measure=1, position=[1, 1, float("inf")])
    )
    assert not base_element.is_approximately_equal(Element(group_id=0, type="annular", dim=2, active=True, measure=1, position=[1, 1, 1]))


def test_element_is_approximately_equal_nan():
    nan_element = Element(group_id=0, type="rect", dim=2, active=True, measure=1, position=[1, 1, float("nan")])
    assert nan_element.is_approximately_equal(nan_element, equal_nan=True)
    assert not nan_element.is_approximately_equal(nan_element, equal_nan=False)


def test_element_is_approximately_equal_boundary():
    base_element = Element(group_id=0, type="rect", dim=2, active=True, measure=1, position=[1, 1, 1])
    boundary_element = Element(group_id=0, type="rect", dim=2, active=True, measure=1, position=[1, 1, 1 + 1.1e-5])
    assert not base_element.is_approximately_equal(boundary_element)
    assert base_element.is_approximately_equal(boundary_element, rtol=1.2e-5)


def test_element_is_approximately_equal_consistency():
    base_element = Element(group_id=0, type="rect", dim=2, active=True, measure=1, position=[1, 1, 1])
    other_element = Element(group_id=0, type="rect", dim=2, active=True, measure=1, position=[1, 1, 1.000001])
    assert base_element.is_approximately_equal(other_element) == np.allclose(base_element.position, other_element.position)
    assert base_element.is_approximately_equal(other_element) == base_element.is_approximately_equal(
        other_element, rtol=1e-05, atol=1e-08, equal_nan=False
    )


def test_element_is_approximately_equal_type_error():
    element = Element(group_id=0, type="rect", dim=2, active=True, measure=1, position=[1, 1, 1])

    with pytest.raises(TypeError, match=r"not an element with <class 'str'> is not of type Element"):
        element.is_approximately_equal("not an element")
