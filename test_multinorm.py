"""Tests for multinorm, using pyest.
"""
from __future__ import division
import warnings
import pytest
import numpy as np
from numpy.testing import assert_allclose
from multinorm import MultiNorm, Parameter


@pytest.fixture()
def mn1():
    """Example test case without correlations."""
    mean = [10, 20, 30]
    covariance = [[1, 0, 0], [0, 4, 0], [0, 0, 9]]
    names = ["a", "b", "c"]
    return MultiNorm(mean, covariance, names)


@pytest.fixture()
def mn2():
    """Example test case with correlations.

    These are pretty much arbitrary numbers,
    when used in tests outputs will only establish
    current behaviour or allow testing against other codes.

    This is the same test case as used in scipy here:
    https://github.com/scipy/scipy/blob/23a648dd5a99de93901abb431885948cd4df14ff/scipy/stats/tests/test_multivariate.py#L349-L350
    """
    mean = [1, 3, 2]
    cov = [[1, 2, 0], [2, 5, 0.5], [0, 0.5, 3]]
    names = ["a", "b", "c"]
    return MultiNorm(mean, cov, names)


def test_init():
    mn = MultiNorm(mean=[1, 2])
    assert mn.names == ["par_0", "par_1"]

    mn = MultiNorm(cov=[[1, 0], [0, 1]])
    assert mn.names == ["par_0", "par_1"]

    mn = MultiNorm(names=["a"])
    assert mn.mean.shape == (1,)
    assert mn.cov.shape == (1, 1)

    mn = MultiNorm()
    assert mn.names == ["par_0"]

    with pytest.raises(ValueError):
        MultiNorm(mean=[0, 0, 0], names=["a", "b"])


def test_init_singular():
    # It should be possible to create a MultiNorm
    # with a singular cov matrix, because e.g.
    # when fixing parameters, that is what comes out.
    cov = [[1, 0], [0, 0]]
    mn = MultiNorm(cov=cov)
    assert_allclose(mn.cov, cov)


def test_repr(mn1):
    assert repr(mn1) == "MultiNorm(n=3)"


def test_str(mn1):
    assert "MultiNorm" in str(mn1)


def test_getitem(mn1):
    par = mn1["c"]

    assert isinstance(par, Parameter)
    assert par.index == 2
    assert par.name == "c"
    assert_allclose(par.mean, 30)
    assert_allclose(par.err, 3)
    assert repr(par) == "Parameter(index=2, name='c')"


def test_err(mn1):
    assert_allclose(mn1.err, [1, 2, 3])


def test_correlation(mn1):
    expected = np.eye(3)
    assert_allclose(mn1.correlation, expected)


def test_precision(mn1):
    expected = np.diag([1, 1 / 4, 1 / 9])
    assert_allclose(mn1.precision, expected)


def test_from_err():
    mean = [10, 20, 30]
    err = [1, 2, 3]
    names = ["a", "b", "c"]
    mn = MultiNorm.from_err(mean, err, names)
    assert_allclose(mn.mean, mean)
    assert_allclose(mn.err, err)
    assert mn.names == names


def test_from_points():
    points = [(10, 20, 30), (12, 20, 30)]
    names = ["a", "b", "c"]
    mn = MultiNorm.from_points(points, names)

    assert mn.names == names
    assert_allclose(mn.mean, [11, 20, 30])
    assert_allclose(mn.cov, [[2, 0, 0], [0, 0, 0], [0, 0, 0]])


def test_from_product():
    mn1 = MultiNorm(mean=[0, 0], names=["a", "b"])
    mn2 = MultiNorm(mean=[2, 4], names=["a", "b"])

    mn = MultiNorm.from_product([mn1, mn2])

    assert mn.names == ["a", "b"]
    assert_allclose(mn.mean, [1, 2])
    assert_allclose(mn.cov, [[0.5, 0], [0, 0.5]])


def test_marginal(mn1):
    mn2 = mn1.marginal([0, 2])
    assert mn2.names == ["a", "c"]
    assert_allclose(mn2.mean, [10, 30])
    assert_allclose(mn2.err, [1, 3])


def test_conditional(mn1):
    mn2 = mn1.conditional(1, 20)
    assert mn2.names == ["a", "c"]
    assert_allclose(mn2.mean, [10, 30])
    assert_allclose(mn2.err, [1, 3])


def test_fix(mn1):
    mn2 = mn1.fix(1)
    assert mn2.names == ["a", "c"]
    assert_allclose(mn2.mean, [10, 30])
    assert_allclose(mn2.err, [1, 3])


def test_sigma_distance(mn1):
    d = mn1.sigma_distance([10, 20, 30])
    assert_allclose(d, 0)

    d = mn1.sigma_distance([10, 20, 33])
    assert_allclose(d, 1)


def test_pdf(mn1):
    res = mn1.pdf([[10, 20, 30]])
    assert_allclose(res, 0.010582272655706831)


def test_logpdf(mn1):
    res = mn1.logpdf([[10, 20, 30]])
    assert_allclose(res, -4.548575068842073)


def test_sample(mn1):
    res = mn1.sample(size=1, random_state=0)
    assert_allclose(res, [10.978738, 20.800314, 35.292157])


def test_pandas_summary(mn1):
    df = mn1.pandas_summary
    # TODO: add asserts


def test_pandas_cov(mn1):
    df = mn1.pandas_cov
    # TODO: add asserts


def test_pandas_correlation(mn1):
    df = mn1.pandas_correlation
    # TODO: add asserts


def test_to_uncertainties(mn1):
    # TODO: remove warning filter when this is resolved:
    # https://github.com/lebigot/uncertainties/pull/88
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        res = mn1.to_uncertainties()

    assert isinstance(res, tuple)
    assert len(res) == 3

    a, b, c = res
    assert_allclose(a.nominal_value, 10)
    assert_allclose(a.std_dev, 1)
    assert_allclose(b.nominal_value, 20)
    assert_allclose(b.std_dev, 2)
    assert_allclose(c.nominal_value, 30)
    assert_allclose(c.std_dev, 3)


def test_to_matplotlib_ellipse(mn1):
    ellipse = mn1.marginal(["a", "b"]).to_matplotlib_ellipse()
    assert_allclose(ellipse.center, (10, 20))
    assert_allclose(ellipse.width, 2)
    assert_allclose(ellipse.height, 4)
    # TODO: change to a test case where `angle` is nonzero and well defined.
    assert_allclose(ellipse.angle, 0)
