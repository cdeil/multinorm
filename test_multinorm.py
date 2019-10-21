"""Tests for multinorm, using pyest.
"""
import warnings
import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from multinorm import MultiNorm


def assert_multinormal_allclose(a, b):
    """Assert that two `MultiNorm` objects are allclose."""
    assert a.names == b.names
    assert_allclose(a.mean, b.mean)
    assert_allclose(a.cov, b.cov)


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

    - [0, 1] = ["a", "b"] correlation: 0.89442719 (large correlation)
    - [0, 2] = ["a", "c"] correlation: 0.0 (uncorrelated)
    - [1, 2] = ["b", "c"] correlation: 0.12909944 (small correlation)

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


@pytest.fixture()
def mn3():
    """Example test case with very large and small numbers.

    This is numerically challenging for several computations.
    Ideally all methods in `MultiNorm` should be able to handle
    this case and give accurate results, because it's quite common.
    """
    mean = np.array([1e-10, 1, 1e10])
    err = 1.0 * mean
    names = ["a", "b", "c"]
    return MultiNorm.from_err(mean, err, names=names)


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


def test_str(mn1):
    assert str(mn1) == """\
MultiNorm with n=3 parameters:
      mean  err
name           
a     10.0  1.0
b     20.0  2.0
c     30.0  3.0"""


def test_eq(mn1):
    assert mn1 == mn1
    assert mn1 != mn2
    assert mn1 != "asdf"


def test_from_err():
    # Test with default: no correlation
    mean = [10, 20, 30]
    err = [1, 2, 3]
    correlation = None
    names = ["a", "b", "c"]
    mn = MultiNorm.from_err(mean, err, correlation, names)
    assert_allclose(mn.mean, mean)
    assert_allclose(mn.cov, [[1, 0, 0], [0, 4, 0], [0, 0, 9]])
    assert mn.names == names

    # Test with given correlation
    correlation = [[1, 0.8, 0], [0.8, 1, 0.1], [0.0, 0.1, 1]]
    mn = MultiNorm.from_err(err=err, correlation=correlation)
    assert_allclose(mn.correlation, correlation)


def test_from_samples():
    points = [(10, 20, 30), (12, 20, 30)]
    names = ["a", "b", "c"]
    mn = MultiNorm.from_samples(points, names)

    assert mn.names == names
    assert_allclose(mn.mean, [11, 20, 30])
    assert_allclose(mn.cov, [[2, 0, 0], [0, 0, 0], [0, 0, 0]])


def test_from_product():
    d1 = MultiNorm(mean=[0, 0], names=["a", "b"])
    d2 = MultiNorm(mean=[2, 4], names=["a", "b"])

    mn = MultiNorm.from_product([d1, d2])

    assert mn.names == ["a", "b"]
    assert_allclose(mn.mean, [1, 2])
    assert_allclose(mn.cov, [[0.5, 0], [0, 0.5]])


def test_from_product_numerics(mn3):
    # Check that `from_product` gives good results
    # even if some parameters are very small and others very large
    # Product of 10 measurements should reduce variance by factor 10
    mn = MultiNorm.from_product([mn3] * 10)
    assert_allclose(mn.cov, np.diag([1e-21, 0.1, 1e19]))


def test_make_example():
    mn = MultiNorm.make_example(n_par=2, n_fix=1, random_state=0)
    assert_allclose(mn.mean, [1.76405235, 0.40015721, 0.97873798])
    expected = [[8.50937518, -0.41563014, 0], [-0.41563014, 1.85774006, 0], [0, 0, 0]]
    assert_allclose(mn.cov, expected)


def test_parameters(mn1):
    df = mn1.parameters

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["mean", "err"]
    assert list(df.index) == ["a", "b", "c"]

    # Access infos per parameter as Series
    par = df.loc["c"]
    assert isinstance(par, pd.Series)
    assert_allclose(par["mean"], 30)
    assert_allclose(par["err"], 3)

    # Access infos per quantity as Series
    mean = df["mean"]
    assert isinstance(mean, pd.Series)
    assert_allclose(mean["c"], 30)


def test_err(mn1, mn2):
    assert_allclose(mn1.err, [1, 2, 3])
    assert_allclose(mn2.err, [1.0, 2.23606798, 1.73205081])


def test_correlation(mn1, mn2):
    expected = np.eye(3)
    assert_allclose(mn1.correlation.values, expected)

    c = mn2.correlation
    assert_allclose(c.iloc[0, 1], 0.89442719)
    assert_allclose(c.iloc[0, 2], 0)
    assert_allclose(c.iloc[1, 2], 0.12909944)


def test_precision(mn1, mn2, mn3):
    expected = np.diag([1, 1 / 4, 1 / 9])
    assert_allclose(mn1.precision, expected)

    expected = [
        [5.36363636, -2.18181818, 0.36363636],
        [-2.18181818, 1.09090909, -0.18181818],
        [0.36363636, -0.18181818, 0.36363636],
    ]
    assert_allclose(mn2.precision, expected)

    expected = np.diag([1e20, 1, 1e-20])
    assert_allclose(mn3.precision, expected)


def test_drop(mn1):
    assert mn1.drop("b") == mn1.marginal(["a", "c"])


def test_marginal(mn1, mn2):
    # Marginal distribution: subset of `cov`
    mn = mn1.marginal([0, 2])
    assert mn.names == ["a", "c"]
    assert_allclose(mn.mean, [10, 30])
    assert_allclose(mn.cov, [[1, 0], [0, 9]])

    mn = mn2.marginal([0, 2])
    assert mn.names == ["a", "c"]
    assert_allclose(mn.mean, [1, 2])
    assert_allclose(mn.cov, [[1, 0], [0, 3]])


def test_conditional(mn1, mn2):
    mn = mn1.conditional(1, 20)
    assert mn.names == ["a", "c"]
    assert_allclose(mn.mean, [10, 30])
    assert_allclose(mn.cov, [[1, 0], [0, 9]])

    mn = mn2.conditional(1, 3)
    assert mn.names == ["a", "c"]
    assert_allclose(mn.mean, [1, 2])
    assert_allclose(mn.cov, [[0.2, -0.2], [-0.2, 2.95]])


def test_fix(mn1, mn2, mn3):
    mn = mn1.fix(1)
    assert mn.names == ["a", "c"]
    assert_allclose(mn.mean, [10, 30])
    assert_allclose(mn.cov, [[1, 0], [0, 9]])

    mn = mn2.fix(1)
    assert mn.names == ["a", "c"]
    assert_allclose(mn.mean, [1, 2])
    assert_allclose(mn.cov, [[0.2, -0.2], [-0.2, 2.95]])
    expected = [[5.363636, 0.363636], [0.363636, 0.363636]]
    assert_allclose(mn.precision, expected, atol=1e-5)

    mn = mn3.fix(1)
    assert_allclose(mn.mean, [1e-10, 1e10])
    expected = np.diag([1e-20, 1e20])
    assert_allclose(mn.cov, expected)


def test_conditional_vs_fix():
    # Conditional and fix should be the same (up to numerical errors)
    n_par = 5
    mn = MultiNorm.make_example(n_par=n_par)

    a = mn.conditional([1, 2, 3])
    b = mn.fix([1, 2, 3])

    assert_multinormal_allclose(a, b)


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


def test_to_matplotlib_ellipse(mn1, mn2):
    ellipse = mn1.marginal(["a", "b"]).to_matplotlib_ellipse()
    assert_allclose(ellipse.center, (10, 20))
    assert_allclose(ellipse.width, 2)
    assert_allclose(ellipse.height, 4)
    # Angle could be 0 or equivalent 180 due to rounding
    angle = np.abs(ellipse.angle - np.array([0, 180])).min()
    assert_allclose(angle, 0)

    ellipse = mn2.marginal(["a", "b"]).to_matplotlib_ellipse()
    assert_allclose(ellipse.center, (1, 3))
    assert_allclose(ellipse.width, 0.82842712)
    assert_allclose(ellipse.height, 4.82842712)
    assert_allclose(ellipse.angle, 157.5)

    with pytest.raises(ValueError):
        mn1.to_matplotlib_ellipse()

def test_to_xarray(mn1):
    data = mn1.to_xarray("pdf")
    assert data.dims == ("a", "b", "c")
    assert_allclose(data.values[1, 2, 3], 4.20932837e-08)
