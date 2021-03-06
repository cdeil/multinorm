"""Tests for multinorm, using pyest."""
import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_equal
from multinorm import MultiNorm


def assert_multinorm_allclose(a, b):
    """Assert that two `MultiNorm` objects are allclose."""
    assert_allclose(a.mean, b.mean)
    assert_allclose(a.cov, b.cov)


@pytest.fixture()
def mn1():
    """Example test case without correlations."""
    mean = [10, 20, 30]
    covariance = [[1, 0, 0], [0, 4, 0], [0, 0, 9]]
    return MultiNorm(mean, covariance)


@pytest.fixture()
def mn2():
    """Example test case with correlations.

    - [0, 1] correlation: 0.89442719 (large correlation)
    - [0, 2] correlation: 0.0 (uncorrelated)
    - [1, 2] correlation: 0.12909944 (small correlation)

    These are pretty much arbitrary numbers,
    when used in tests outputs will only establish
    current behaviour or allow testing against other codes.

    This is the same test case as used in scipy here:
    https://github.com/scipy/scipy/blob/23a648dd5a99de93901abb431885948cd4df14ff/scipy/stats/tests/test_multivariate.py#L349-L350
    """
    mean = [1, 3, 2]
    cov = [[1, 2, 0], [2, 5, 0.5], [0, 0.5, 3]]
    return MultiNorm(mean, cov)


@pytest.fixture()
def mn3():
    """Example test case with very large and small numbers.

    This is numerically challenging for several computations.
    Ideally all methods in `MultiNorm` should be able to handle
    this case and give accurate results, because it's quite common.
    """
    mean = np.array([1e-10, 1, 1e10])
    error = 1.0 * mean
    return MultiNorm.from_error(mean, error)


@pytest.mark.parametrize("dtype", ["int8", "float32"])
def test_init_dtype(dtype):
    # Make sure we always get float64 and good precision
    mn = MultiNorm(np.zeros(2, dtype=dtype), np.eye(2, dtype=dtype))
    assert mn.mean.dtype == np.float64
    assert mn.cov.dtype == np.float64


def test_init_partially():
    mn = MultiNorm(mean=[1, 2])
    assert_allclose(mn.error, [1, 1])

    mn = MultiNorm(cov=[[1, 0], [0, 1]])
    assert_allclose(mn.mean, [0, 0])

def test_init_empty():
    mn = MultiNorm()
    assert mn.mean.shape == (1,)
    assert mn.cov.shape == (1, 1)

def test_init_bad():
    with pytest.raises(ValueError):
        MultiNorm(mean=[0, 0, 0], cov=[[1, 2], [3, 4]])


def test_init_singular():
    # It should be possible to create a MultiNorm
    # with a singular cov matrix, because e.g.
    # when fixing parameters, that is what comes out.
    cov = [[1, 0], [0, 0]]
    mn = MultiNorm(cov=cov)
    assert_allclose(mn.cov, cov)


def test_str(mn1):
    expected = """\
MultiNorm with n=3 parameters:
   mean  error
0  10.0    1.0
1  20.0    2.0
2  30.0    3.0"""
    assert str(mn1) == expected


def test_eq(mn1):
    assert mn1 == mn1
    assert mn1 != mn2
    assert mn1 != "asdf"


def test_from_err():
    # Test with default: no correlation
    mean = [10, 20, 30]
    err = [1, 2, 3]
    correlation = None
    mn = MultiNorm.from_error(mean, err, correlation)
    assert_allclose(mn.mean, mean)
    assert_allclose(mn.cov, [[1, 0, 0], [0, 4, 0], [0, 0, 9]])

    # Test with given correlation
    correlation = [[1, 0.8, 0], [0.8, 1, 0.1], [0.0, 0.1, 1]]
    mn = MultiNorm.from_error(error=err, correlation=correlation)
    assert_allclose(mn.correlation, correlation)

    with pytest.raises(ValueError):
        MultiNorm.from_error(mean)


def test_from_samples():
    points = [(10, 20, 30), (12, 20, 30)]
    mn = MultiNorm.from_samples(points)

    assert_allclose(mn.mean, [11, 20, 30])
    assert_allclose(mn.cov, [[2, 0, 0], [0, 0, 0], [0, 0, 0]])


def test_from_stack():
    d1 = MultiNorm(mean=[1, 2], cov=np.full((2, 2), 2))
    d2 = MultiNorm(mean=[3, 4, 5], cov=np.full((3, 3), 3))

    mn = MultiNorm.from_stack([d1, d2])

    assert_allclose(mn.mean, [1, 2, 3, 4, 5])
    assert_allclose(mn.cov[0], [2, 2, 0, 0, 0])
    assert_allclose(mn.cov[4], [0, 0, 3, 3, 3])


def test_from_product():
    d1 = MultiNorm(mean=[0, 0])
    d2 = MultiNorm(mean=[2, 4])

    mn = MultiNorm.from_product([d1, d2])

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


def test_summary_dataframe(mn1):
    df = mn1.summary_dataframe(n_sigma=1)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["mean", "error", "lo", "hi"]
    assert list(df.index) == [0, 1, 2]

    # Access infos per parameter as Series
    par = df.iloc[2]
    assert isinstance(par, pd.Series)
    assert_allclose(par["mean"], 30)
    assert_allclose(par["error"], 3)
    assert_allclose(par["lo"], 27)
    assert_allclose(par["hi"], 33)

    # Access infos per quantity as Series
    mean = df["mean"]
    assert isinstance(mean, pd.Series)
    assert_allclose(mean[2], 30)


def test_error(mn1, mn2):
    error = mn1.error
    assert isinstance(error, np.ndarray)
    assert error.shape == (3,)
    assert_allclose(error, [1, 2, 3])

    assert_allclose(mn2.error, [1.0, 2.23606798, 1.73205081])


def test_correlation(mn1, mn2):
    expected = np.eye(3)
    assert_allclose(mn1.correlation, expected)

    c = mn2.correlation
    assert_allclose(c[0, 1], 0.89442719)
    assert_allclose(c[0, 2], 0)
    assert_allclose(c[1, 2], 0.12909944)


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


def test_marginal(mn1, mn2):
    mn = mn1.marginal([0, 2])
    assert_allclose(mn.mean, [10, 30])
    assert_allclose(mn.cov, [[1, 0], [0, 9]])

    mn = mn2.marginal([0, 2])
    assert_allclose(mn.mean, [1, 2])
    assert_allclose(mn.cov, [[1, 0], [0, 3]])


def test_conditional(mn1, mn2):
    mn = mn1.conditional(1, 20)
    assert_allclose(mn.mean, [10, 30])
    assert_allclose(mn.cov, [[1, 0], [0, 9]])

    mn = mn2.conditional(1, 3)
    assert_allclose(mn.mean, [1, 2])
    assert_allclose(mn.cov, [[0.2, -0.2], [-0.2, 2.95]])


def test_fix(mn1, mn2, mn3):
    mn = mn1.fix(1)
    assert_allclose(mn.mean, [10, 30])
    assert_allclose(mn.cov, [[1, 0], [0, 9]])

    mn = mn2.fix(1)
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
    mn = MultiNorm.make_example(n_par=5)

    a = mn.conditional([1, 2, 3])
    b = mn.fix([1, 2, 3])

    assert_multinorm_allclose(a, b)


def test_sigma_distance(mn1):
    res = mn1.sigma_distance([10, 20, 30])
    assert res.shape == ()
    assert_allclose(res, 0)

    # Multiple points at once should work
    res = mn1.sigma_distance([[10, 20, 30], [10, 20, 33]])
    assert res.shape == (2,)
    assert_allclose(res, [0, 1])


def test_pdf(mn1):
    res = mn1.pdf([[10, 20, 30]])
    assert res.shape == ()
    assert_allclose(res, 0.010582272655706831)

    # Multiple points at once should work
    res = mn1.pdf([[10, 20, 30], [10, 20, 30]])
    assert res.shape == (2,)


def test_logpdf(mn1):
    res = mn1.logpdf([[10, 20, 30]])
    assert_allclose(res, -4.548575068842073)


def test_sample(mn1):
    res = mn1.sample(size=1, random_state=0)
    assert_allclose(res, [[10.978738, 20.800314, 35.292157]])
    assert res.shape == (1, 3)

    res = mn1.sample(size=2, random_state=0)
    assert res.shape == (2, 3)


def test_to_uncertainties(mn1):
    uncertainties = pytest.importorskip("uncertainties")

    res = mn1.to_uncertainties()
    a, b, c = res

    assert isinstance(res, tuple)
    assert len(res) == 3

    assert isinstance(a, uncertainties.core.AffineScalarFunc)
    assert_allclose(a.nominal_value, 10)
    assert_allclose(a.std_dev, 1)

    assert isinstance(b, uncertainties.core.AffineScalarFunc)
    assert_allclose(b.nominal_value, 20)
    assert_allclose(b.std_dev, 2)

    assert isinstance(c, uncertainties.core.AffineScalarFunc)
    assert_allclose(c.nominal_value, 30)
    assert_allclose(c.std_dev, 3)


def test_error_ellipse(mn2):
    ellipse = mn2.marginal([0, 1]).error_ellipse()
    assert_allclose(ellipse["xy"], (1, 3))
    assert_allclose(ellipse["width"], 0.82842712)
    assert_allclose(ellipse["height"], 4.82842712)
    assert_allclose(ellipse["angle"], 157.5)


def test_to_matplotlib_ellipse(mn1, mn2):
    ellipse = mn1.marginal([0, 1]).to_matplotlib_ellipse()
    assert_allclose(ellipse.center, (10, 20))
    assert_allclose(ellipse.width, 2)
    assert_allclose(ellipse.height, 4)
    # Angle could be 0 or equivalent 180 due to rounding
    angle = np.abs(ellipse.angle - np.array([0, 180])).min()
    assert_allclose(angle, 0)

    ellipse = mn2.marginal([0, 1]).to_matplotlib_ellipse()
    assert_allclose(ellipse.center, (1, 3))
    assert_allclose(ellipse.width, 0.82842712)
    assert_allclose(ellipse.height, 4.82842712)
    assert_allclose(ellipse.angle, 157.5)

    with pytest.raises(ValueError):
        mn1.to_matplotlib_ellipse()


def test_to_xarray(mn1):
    data = mn1.to_xarray("pdf")
    # TODO: assert on all properties
    assert_allclose(data.values[1, 2, 3], 4.20932837e-08)


def test_make_index_mask(mn1):
    assert_equal(mn1.make_index_mask([0, 2]), [True, False, True])
    assert_equal(mn1.make_index_mask([True, False, True]), [True, False, True])
