# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
multinorm - Multivariate Normal Distributions for Humans.

A Python class to work with model fit results
(parameters and the covariance matrix).

- Code: https://github.com/cdeil/multinorm
- Docs: https://multinorm.readthedocs.io
- License: BSD-3-Clause
"""
from __future__ import division
from pkg_resources import get_distribution, DistributionNotFound
import numpy as np
from scipy.stats import multivariate_normal

__all__ = ["MultiNorm"]

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass


class MultiNorm(object):
    """Multivariate normal distribution.

    Given ``n`` parameters, the ``mean`` and ``names``
    should be one-dimensional with size ``n``,
    and ``cov`` should be a two-dimensional matrix of shape ``(n, n)``.

    Documentation for this class:

    - Tutorial Jupyter notebook: `multinorm.ipynb`_
    - Documentation: :ref:`gs`, :ref:`create`, :ref:`analyse`
    - Equations and statistics: :ref:`theory`

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

        # For other cached properties
        self._cache = {}

    def __repr__(self):
        return "{}(n={})".format(self.__class__.__name__, self.n)

    def __str__(self):
        s = repr(self)

        s += "\nnames: "
        s += str(self.names)

        s += "\nmean: "
        s += str(self.mean)

        s += "\nerr: "
        s += str(self.err)

        s += "\ncov:\n"
        s += str(self.cov)

        return s

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

        precisions = [_.precision for _ in distributions]
        precision = np.sum(precisions, axis=0)
        cov = cls._cov_from_precision(precision)

        means_weighted = [_._mean_weighted for _ in distributions]
        means_weighted = np.sum(means_weighted, axis=0)
        mean = np.dot(cov, means_weighted)
        return cls(mean, cov, names)

    @classmethod
    def make_example(cls, n_par, n_fix=0, random_state=None):
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
            Seed (int), or ``None`` to choose random seed.
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

    @property
    def pandas_summary(self):
        """Summary table (`pandas.DataFrame`).

        TODO: choose a good name for this
        TODO: document or link to docs
        """
        import pandas as pd

        data = {"mean": self.mean, "err": self.err}
        index = pd.Index(self.names, name="name")
        return pd.DataFrame(data, index)

    @property
    def pandas_cov(self):
        """Covariance matrix (`pandas.DataFrame`)."""
        return self._pandas_matrix(self.cov)

    @property
    def pandas_correlation(self):
        """Correlation matrix (`pandas.DataFrame`)."""
        return self._pandas_matrix(self.correlation)

    def _pandas_matrix(self, matrix):
        import pandas as pd

        index = pd.Index(self.names, name="name")
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

        return correlated_values(self.mean, self.cov)

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

        # See https://stackoverflow.com/questions/12301071
        xy = self.mean

        vals, vecs = self._eigh
        width, height = 2 * n_sigma * np.sqrt(vals)
        angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

        return Ellipse(xy=xy, width=width, height=height, angle=angle, **kwargs)

    @property
    def _eigh(self):
        # TODO: can we replace this with something from `self.scipy.cov_info`?
        return np.linalg.eigh(self.cov)

    @staticmethod
    def _cov_from_precision(precision):
        # TODO: is there a better (faster or more accurate) way to compute `cov`?
        return np.linalg.inv(precision)

    @property
    def _mean_weighted(self):
        return np.dot(self.precision, self.mean)

    @property
    def n(self):
        """Number of dimensions of the distribution (int).

        Given by the number of parameters.
        """
        return self.scipy.dim

    @property
    def mean(self):
        """Mean vector (`numpy.ndarray`)."""
        return self.scipy.mean

    @property
    def cov(self):
        """Covariance matrix (`numpy.ndarray`)."""
        return self.scipy.cov

    @property
    def names(self):
        """Parameter names (`list` of `str`)."""
        return self._name_index.names

    @property
    def err(self):
        r"""Error vector (`numpy.ndarray`).

        Defined as :math:`\sigma_i = \sqrt{\Sigma_{ii}}`.
        """
        if "err" not in self._cache:
            self._cache["err"] = np.sqrt(np.diag(self.cov))

        return self._cache["err"]

    @property
    def correlation(self):
        r"""Correlation matrix (:class:`numpy.ndarray`).

        Correlation :math:`C` is related to covariance :math:`\Sigma` via:

        .. math::
            C_{ij} = \frac{ \Sigma_{ij} }{ \sqrt{\Sigma_{ii} \Sigma_{jj}} }
        """
        if "correlation" not in self._cache:
            c = self.cov / np.outer(self.err, self.err)
            self._cache["correlation"] = c

        return self._cache["correlation"]

    @property
    def precision(self):
        """Precision matrix (`numpy.ndarray`).

        The inverse of the covariance matrix.

        Sometimes called the "information matrix" or "Hesse matrix".
        """
        return self.scipy.cov_info.pinv

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
        names = self._name_index.get_names(mask)

        mean = self.mean[mask]
        cov = self.cov[np.ix_(mask, mask)]
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
            values = self.mean[mask2]
        else:
            values = np.asarray(values, dtype=float)

        mean1 = self.mean[mask1]
        mean2 = self.mean[mask2]

        cov11 = self.cov[np.ix_(mask1, mask1)]
        cov12 = self.cov[np.ix_(mask1, mask2)]
        cov21 = self.cov[np.ix_(mask2, mask1)]
        cov22 = self.cov[np.ix_(mask2, mask2)]

        # TODO: would it be better to compute the inverse of cov22
        # instead of calling solve twice?
        mean = mean1 + np.dot(cov12, np.linalg.solve(cov22, values - mean2))
        cov = cov11 - np.dot(cov12, np.linalg.solve(cov22, cov21))

        return self.__class__(mean, cov, names)

    def fix(self, pars):
        """Fix some parameters.

        See :ref:`theory_fix`.

        Parameters
        ----------
        pars : list
            Parameters to fix (indices or names)
        """
        # mask of parameters to keep (that are not fixed)
        mask = np.invert(self._name_index.get_mask(pars))
        names = self._name_index.get_names(mask)

        mean = self.mean[mask]
        precision = self.precision[np.ix_(mask, mask)]
        cov = self._cov_from_precision(precision)
        return self.__class__(mean, cov, names)

    def sigma_distance(self, point):
        """Number of standard deviations from the mean (float).

        Also called the Mahalanobis distance.
        See :ref:`theory_sigmas`.
        """
        point = np.asanyarray(point)
        d = self.mean - point
        sigma = np.dot(np.dot(d.T, self.precision), d)
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

    def __getitem__(self, par):
        idx = self._name_index.get_idx(par)[0]
        return Parameter(idx, self)


class Parameter(object):
    """Parameter information.

    Proxy object to MultiNorm.

    TODO: should we instead pre-compute and cover everything over?
    """

    def __init__(self, index, multi_norm):
        self._index = index
        self._multi_norm = multi_norm

    def __repr__(self):
        return "Parameter(index={!r}, name={!r})".format(self.index, self.name)

    @property
    def index(self):
        return self._index

    @property
    def name(self):
        return self._multi_norm.names[self.index]

    @property
    def mean(self):
        return self._multi_norm.mean[self.index]

    @property
    def err(self):
        return self._multi_norm.err[self.index]


class _NameIndex(object):
    """Parameter index.

    Doesn't do much, just store parameter names
    and match parameter names and indices.

    If we go all-in on pandas, we could use a
    pandas index for this functionality.
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
