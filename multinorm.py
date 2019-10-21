# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
multinorm - Multivariate Normal Distributions from Python.

A Python class to work with model fit results
(parameters and the covariance matrix).

- Code: https://github.com/cdeil/multinorm
- Docs: https://multinorm.readthedocs.io
- License: BSD-3-Clause
"""
from pkg_resources import get_distribution, DistributionNotFound
import functools
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.stats import multivariate_normal

__all__ = ["MultiNorm"]

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass


# Lazy property taken from here:
# https://stackoverflow.com/a/3013910/498873
# In Python 3.8 a functools.cached_property is added
# So we change to that name
def cached_property(fn):
    attr_name = "_cache_" + fn.__name__

    @property
    @functools.wraps(fn)
    def _cached_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _cached_property


def _matrix_inverse(matrix):
    # np.linalg.inv seems to give numerically stable results
    # We need inverse in several places, so in case there's
    # a better option, and to get consistency, we put this wrapper function
    return np.linalg.inv(matrix)


class MultiNorm:
    """Multivariate normal distribution.

    Given ``n`` parameters, the ``mean`` and ``names``
    should be one-dimensional with size ``n``,
    and ``cov`` should be a two-dimensional matrix of shape ``(n, n)``.

    Documentation for this class:

    - Tutorial Jupyter notebook: `multinorm.ipynb`_
    - Documentation: :ref:`gs`, :ref:`create`, :ref:`analyse`
    - Equations and statistics: :ref:`theory`

    Note that MultiNorm objects should be used read-only,
    almost all properties are cached. If you need to modify
    values, make a new `MultiNorm` object.

    Parameters
    ----------
    mean : numpy.ndarray
        Mean vector
    cov : numpy.ndarray
        Covariance matrix
    names : list
        Python list of parameter names (str).
        Default: use "par_i" with ``i = 0 .. N - 1``.
    """

    def __init__(self, mean=None, cov=None, names=None):
        # multivariate_normal does a lot of input validation
        # so we call it first to avoid having to duplicate that
        self._scipy = multivariate_normal(mean, cov, allow_singular=True)
        self._name_index = _NameIndex(names, self.n)

    def __repr__(self):
        s = "{} with n={} parameters:\n".format(self.__class__.__name__, self.n)
        s += str(self.parameters)
        return s

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented

        return (
                self.names == other.names
                and (self.mean == other.mean).all()
                and (self.cov == other.cov).all(axis=None)
        )

    @classmethod
    def from_err(cls, mean=None, err=None, correlation=None, names=None):
        r"""Create `MultiNorm` from parameter errors.

        With errors :math:`\sigma_i` this will create a
        diagonal covariance matrix with

        .. math::
            \Sigma_{ii} = \sigma_i^2

        For a given ``correlation``, or in general: this will create
        a `MultiNormal` with a covariance matrix such that it's
        ``err`` and ``correlation`` match the one specified here
        (up to rounding errors).

        Parameters
        ----------
        mean : numpy.ndarray
            Mean vector
        err : numpy.ndarray
            Error vector
        correlation : numpy.ndarray
            Correlation matrix
        names : list
            Parameter names
        """
        if err is None:
            if mean is None:
                raise ValueError("Must give mean or err")

            err = np.ones_like(mean)

        err = np.asarray(err, dtype=float)
        n = len(err)

        if correlation is None:
            correlation = np.eye(n)
        else:
            correlation = np.asarray(correlation, dtype=float)

        cov = correlation * np.outer(err, err)

        return cls(mean, cov, names)

    @classmethod
    def from_points(cls, points, names=None):
        """Create `MultiNorm` from parameter points.

        Usually the points are samples from some distribution
        and creating this `MultiNorm` distribution is an
        estimate / approximation of that distribution of interest.

        See: :ref:`create_from_points`.

        Parameters
        ----------
        points : numpy.ndarray
            Array of data points with shape ``(n, 2)``.
        """
        mean = np.mean(points, axis=0)
        cov = np.cov(points, rowvar=False)
        return cls(mean, cov, names)

    @classmethod
    def from_product(cls, distributions):
        """Create `MultiNorm` as product distribution.

        This represents the joint likelihood distribution, assuming
        the individual distributions are from independent measurements.

        See :ref:`theory_product` .

        Parameters
        ----------
        distributions : list
            Python list of `MultiNorm` distributions.

        Returns
        -------
        MultiNorm
            Product distribution
        """
        names = distributions[0].names

        precisions = [_.precision.values for _ in distributions]
        precision = np.sum(precisions, axis=0)
        cov = _matrix_inverse(precision)

        means_weighted = [_._mean_weighted for _ in distributions]
        means_weighted = np.sum(means_weighted, axis=0)
        mean = np.dot(cov, means_weighted)
        return cls(mean, cov, names)

    @classmethod
    def make_example(cls, n_par=3, n_fix=0, random_state=42):
        """Create example `MultiNorm` for testing.

        This is a factory method that allows the quick creation
        of example `MultiNormal` with any number of parameters for testing.

        See: :ref:`create_make_example`.

        Parameters
        ----------
        n_par : int
            Number of parameters
        n_fix : int
            Number of fixed parameters
            in addition to ``n_par``.
        random_state :
            Seed (int) - default: 42
            Put ``None`` to choose random seed.
            Can also pass `numpy.random.RandomState` object.
        """
        n = n_par + n_fix
        rng = np.random.RandomState(random_state)
        mean = rng.normal(size=n)

        s = rng.normal(size=(n_par, n_par))
        cov1 = np.dot(s, s.T)

        cov2 = np.zeros((n, n))
        cov2[:n_par, :n_par] = cov1

        return cls(mean, cov2)

    @property
    def scipy(self):
        """Frozen `scipy.stats.multivariate_normal`_ distribution object.

        A cached property. Used for many computations internally.
        """
        return self._scipy

    @cached_property
    def parameters(self):
        """Parameter table (`pandas.DataFrame`).

        Index is "name", columns are "mean" and "err"
        """
        data = {"mean": self.mean, "err": self.err}
        index = pd.Index(self.names, name="name")
        return pd.DataFrame(data, index)

    def confidence_interval(self, n_sigma=1):
        """Confidence interval table (`pandas.DataFrame`).

        Index is "name", columns are "lo" and "hi"
        """
        d = n_sigma * self.err
        data = {"lo": self.mean - d, "hi": self.mean + d}
        index = pd.Index(self.names, name="name")
        return pd.DataFrame(data, index)

    def _pandas_series(self, data, name):
        index = pd.Index(self.names, name="name")
        return pd.Series(data, index, name=name)

    def _pandas_matrix(self, matrix):
        index = pd.Index(self.names, name="name")
        # TODO: use `index` twice or make separate `columns` index?
        columns = pd.Index(self.names, name="name")
        return pd.DataFrame(matrix, index, columns)

    def to_uncertainties(self):
        """Convert to `uncertainties`_ objects.

        A tuple of numbers with uncertainties
        (one for each parameter) is returned.

        The `uncertainties`_ package makes it easy to
        do error propagation on derived quantities.

        See examples in :ref:`analyse`.
        """
        from uncertainties import correlated_values

        return correlated_values(self.scipy.mean, self.scipy.cov, self.names)

    def to_xarray(self, fcn='pdf', n_sigma=3, num=100):
        """Make an `xarray.DataArray` rastered image.

        This is mostly useful for visualisation.

        All computations can be done without this image.

        TODO: document better

        TODO: add "pmf" option with integral probabilities per pixel

        Parameters
        ----------
        fcn : {"pdf", "logpdf", "stat", "sigma"}
            Function to compute data values
        n_sigma : int
            Number of standard deviations. Controls image coordinate range.
        num : int
            Number of pixels in each dimension. Controls image resolution.
        """
        from xarray import DataArray

        coords = [
            np.linspace(row['lo'], row['hi'], num)
            for _, row in self.confidence_interval(n_sigma).iterrows()
        ]
        points = [_.flatten() for _ in np.meshgrid(*coords)]
        points = np.array(points).T

        if fcn == 'pdf':
            data = self.pdf(points)
        elif fcn == 'logpdf':
            data = self.logpdf(points)
        elif fcn == 'stat':
            data = - 2 * self.logpdf(points)
        elif fcn == 'sigma':
            data = self.sigma_distance(points)
        else:
            raise ValueError(f"Invalid fcn: {fcn!r}")

        data = data.reshape(self.n * (num,))

        return DataArray(data, coords, self.names)

    def to_matplotlib_ellipse(self, n_sigma=1, **kwargs):
        """Create `matplotlib.patches.Ellipse`_.

        See examples in :ref:`plot`.

        Parameters
        ----------
        n_sigma : int
            Number of standard deviations. See :ref:`theory_sigmas`.
        """
        if self.n != 2:
            raise ValueError(
                "Ellipse only available for n=2. "
                "To select parameters, call ``marginal`` or ``conditional`` first."
            )

        from matplotlib.patches import Ellipse

        ellipse = self._compute_ellipse(n_sigma)

        return Ellipse(**ellipse, **kwargs)

    def _compute_ellipse(self, n_sigma=1):
        # See https://stackoverflow.com/questions/12301071
        xy = self.scipy.mean
        vals, vecs = self._eigh
        width, height = 2 * n_sigma * np.sqrt(vals)
        angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        return {"xy": xy, "width": width, "height": height, "angle": angle}

    def plot(self, ax=None, n_sigma=1, **kwargs):
        import matplotlib.pyplot as plt
        ax = plt.gca() if ax is None else ax
        ellipse = self.to_matplotlib_ellipse(n_sigma, **kwargs)
        ax.add_artist(ellipse)

    @cached_property
    def _eigh(self):
        # TODO: can this be computed from `self.scipy.cov_info.U`?
        # TODO: expose covar eigenvalues and vectors?
        return eigh(self.scipy.cov)

    @cached_property
    def _mean_weighted(self):
        return np.dot(self.precision.values, self.mean.values)

    @cached_property
    def n(self):
        """Number of dimensions of the distribution (int).

        Given by the number of parameters.
        """
        return self.scipy.dim

    @cached_property
    def mean(self):
        """Mean vector (`pandas.Series`)."""
        return self._pandas_series(self.scipy.mean, "mean")

    @cached_property
    def cov(self):
        """Covariance matrix (`pandas.DataFrame`)."""
        return self._pandas_matrix(self.scipy.cov)

    # TODO: probably should make this a pandas Index.
    @cached_property
    def names(self):
        """Parameter names (`list` of `str`)."""
        return self._name_index.names

    @cached_property
    def _err(self):
        return np.sqrt(np.diag(self.scipy.cov))

    @cached_property
    def err(self):
        r"""Error vector (`pandas.DataFrame`).

        Defined as :math:`\sigma_i = \sqrt{\Sigma_{ii}}`.
        """
        return self._pandas_series(self._err, "err")

    @cached_property
    def correlation(self):
        r"""Correlation matrix (`pandas.DataFrame`).

        Correlation :math:`C` is related to covariance :math:`\Sigma` via:

        .. math::
            C_{ij} = \frac{ \Sigma_{ij} }{ \sqrt{\Sigma_{ii} \Sigma_{jj}} }
        """
        c = self.cov / np.outer(self.err, self.err)
        return self._pandas_matrix(c)

    @cached_property
    def precision(self):
        """Precision matrix (`pandas.DataFrame`).

        The inverse of the covariance matrix.

        Sometimes called the "information matrix" or "Hesse matrix".
        """
        matrix = _matrix_inverse(self.scipy.cov)
        return self._pandas_matrix(matrix)

    def drop(self, pars):
        """Drop parameters.

        This simply removes the entry from the `mean` vector,
        and the corresponding column and row from the `cov` matrix.

        The computation is the same as :meth:`MultiNorm.marginal`,
        only here the parameters to drop are given, and there
        the parameters to keep are given.

        Parameters
        ----------
        pars : list
            Parameters to fix (indices or names)
        """
        mask = np.invert(self._name_index.get_mask(pars))
        return self._subset(mask)

    def marginal(self, pars):
        """Marginal `MultiNormal` distribution.

        See :ref:`theory_marginal`.

        Parameters
        ----------
        pars : list
            List of parameters (integer indices)

        Returns
        -------
        MultiNorm
            Marginal distribution
        """
        mask = self._name_index.get_mask(pars)
        return self._subset(mask)

    def _subset(self, mask):
        names = self._name_index.get_names(mask)

        mean = self.scipy.mean[mask]
        cov = self.scipy.cov[np.ix_(mask, mask)]
        return self.__class__(mean, cov, names)

    def conditional(self, pars, values=None):
        """Conditional `MultiNormal` distribution.

        Resulting lower-dimensional distribution obtained
        by fixing ``pars`` to ``values``. The output
        distribution is for the other parameters, the
        complement of ``pars``.

        See :ref:`theory_conditional`.

        Parameters
        ----------
        pars : list
            Fixed parameters (indices or names)
        values : list
            Fixed parameters (values).
            Default is to use the values from ``mean``.

        Returns
        -------
        MultiNorm
            Conditional distribution
        """
        # The following code follows the formulae from
        # https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions
        # - "2" refers to the fixed parameters
        # - "1" refers to the remaining (kept) parameters
        mask2 = self._name_index.get_mask(pars)
        mask1 = np.invert(mask2)

        names = self._name_index.get_names(mask1)

        if values is None:
            values = self.scipy.mean[mask2]
        else:
            values = np.asarray(values, dtype=float)

        mean1 = self.scipy.mean[mask1]
        mean2 = self.scipy.mean[mask2]

        cov11 = self.scipy.cov[np.ix_(mask1, mask1)]
        cov12 = self.scipy.cov[np.ix_(mask1, mask2)]
        cov21 = self.scipy.cov[np.ix_(mask2, mask1)]
        cov22 = self.scipy.cov[np.ix_(mask2, mask2)]

        # TODO: would it be better to compute the inverse of cov22
        # instead of calling solve twice?
        mean = mean1 + np.dot(cov12, np.linalg.solve(cov22, values - mean2))
        cov = cov11 - np.dot(cov12, np.linalg.solve(cov22, cov21))

        return self.__class__(mean, cov, names)

    def fix(self, pars):
        """Fix parameters.

        See :ref:`theory_fix`.

        Parameters
        ----------
        pars : list
            Parameters to fix (indices or names)
        """
        # mask of parameters to keep (that are not fixed)
        mask = np.invert(self._name_index.get_mask(pars))
        names = self._name_index.get_names(mask)

        mean = self.scipy.mean[mask]
        precision = self.precision.values[np.ix_(mask, mask)]
        cov = _matrix_inverse(precision)
        return self.__class__(mean, cov, names)

    def standardize(self):
        r"""Standardized distribution.

        For a random variable :math:`x` with a distribution with mean :math:`\mu`
        and standard deviation :math:`\sigma`, given by :math:`Z = (x - \mu) / \sigma`,
        so that :math:`Z` has a distribution with mean zero and standard deviation of one.

        Returns a new distribution object with mean zero,
        and covariance matrix given by the ``correlation`` matrix.
        """
        mean = np.zeros(self.n)
        cov = self.scipy.cov / np.outer(self._err, self._err)
        return self.__class__(mean, cov, self.names)

    def sigma_distance(self, point):
        """Number of standard deviations from the mean (float).

        Also called the Mahalanobis distance.
        See :ref:`theory_sigmas`.
        """
        point = np.asanyarray(point)
        d = self.mean.values - point
        sigma = np.dot(np.dot(d.T, self.precision.values), d)
        return np.sqrt(sigma)

    def pdf(self, points):
        """Probability density function.

        Calls `scipy.stats.multivariate_normal`_.
        """
        return self.scipy.pdf(points)

    def logpdf(self, points):
        """Natural log of PDF.

        Calls `scipy.stats.multivariate_normal`_.
        """
        return self.scipy.logpdf(points)

    def sample(self, size=1, random_state=None):
        """Draw random samples.

        Calls `scipy.stats.multivariate_normal`_.
        """
        return self.scipy.rvs(size, random_state)


# TODO: remove, use pandas Index instead somehow
class _NameIndex(object):
    """Parameter index.

    Doesn't do much, just store parameter names
    and match parameter names and indices.
    """

    def __init__(self, names, n):
        if names is None:
            names = ["par_{}".format(idx) for idx in range(n)]
        else:
            if len(names) != n:
                raise ValueError("len(names) = {}, expected n={}".format(len(names), n))

        self.names = names

    @property
    def n(self):
        return len(self.names)

    def get_idx(self, pars):
        """Create parameter index array.

        Supports scalar, list and array input pars
        Support parameter indices (int) and names (str)
        """
        # TODO: should we support scalar?
        if isinstance(pars, (int, str)):
            pars = [pars]

        idxs = []
        for par in pars:
            # TODO: improve type handling (make str work on Python 2)
            # and give good error messages
            # Probably move this to a separate method.
            if isinstance(par, int):
                idxs.append(par)
            elif isinstance(par, str):
                idx = self.names.index(par)
                idxs.append(idx)
            else:
                raise TypeError()

        return np.asarray(idxs, dtype=int)

    def get_mask(self, pars):
        idx = self.get_idx(pars)
        mask = np.zeros(self.n, dtype=bool)
        mask[idx] = True
        return mask

    def get_names(self, selection):
        # This works for an index array or mask for the selection
        return list(np.array(self.names)[selection])
        # See https://docs.python.org/3.1/library/itertools.html#itertools.compress
        # return [d for d, s in zip(self.names, mask) if s]
