# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
multinorm - Multivariate Normal Distributions for Humans.

A Python class to work with model fit results
(parameters and the covariance matrix).

- Code: https://github.com/cdeil/multinorm
- Docs: https://multinorm.readthedocs.io
- License: BSD-3-Clause
"""
from pkg_resources import get_distribution, DistributionNotFound
import numpy as np
import scipy.stats
import scipy.linalg

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

    See the documentation.

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
        n = _get_n_from_inputs(mean, cov, names)

        if mean is None:
            mean = np.zeros(n, dtype=float)
        else:
            mean = np.asarray(mean, dtype=float)
            if mean.shape != (n,):
                raise ValueError(
                    "mean shape = {!r}, expected ({},)".format(mean.shape, n)
                )

        if cov is None:
            cov = np.eye(n, dtype=float)
        else:
            cov = np.asarray(cov, dtype=float)
            if cov.shape != (n, n):
                raise ValueError(
                    "cov shape = {!r}, expected ({}, {})".format(cov.shape, n, n)
                )

        if names is None:
            names = ["par_{}".format(idx) for idx in range(n)]
        else:
            if len(names) != n:
                raise ValueError("len(names) = {}, expected n={}".format(len(names), n))

        self._n = n
        self._mean = mean
        self._cov = cov
        self._names = names

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

    # TODO: add `corr=None` option to optionally set correlations
    @classmethod
    def from_err(cls, mean=None, err=None, names=None):
        r"""Create `MultiNorm` from parameter errors.

        With errors :math:`\sigma_i` this will create a
        diagonal covariance matrix with

        .. math::
            \Sigma_{ii} = \sigma_i^2
        """
        err = np.asarray(err, dtype=float)
        cov = np.diag(np.power(err, 2))
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
        mean = np.mean(points, axis=None)
        cov = np.cov(points, rowvar=False)
        return cls(mean, cov, names)

    @classmethod
    def joint(cls, distributions):
        """Create joint `MultiNorm` distribution.

        See :ref:`theory_combine` .

        Parameters
        ----------
        distributions : list
            Python list of `MultiNorm` distributions.

        Returns
        -------
        MultiNorm
            Combined joint distribution
        """
        names = distributions[0].names

        precisions = [_.precision for _ in distributions]
        precision = np.sum(precisions, axis=0)
        cov = np.linalg.inv(precision)

        means_weighted = [_._mean_weighted for _ in distributions]
        means_weighted = np.sum(means_weighted, axis=0)
        mean = np.dot(cov, means_weighted)
        return cls(mean, cov, names)

    def to_scipy(self):
        """Convert to `scipy.stats.multivariate_normal`_ object.

        The returned object is a "frozen" distribution object,
        with ``mean`` and ``covar`` set. It offers methods
        for ``pdf``, ``logpdf`` to evaluate the probability density
        function at given points, and ``rvs`` to draw random variate
        samples, i.e. random points from the distribution.

        See examples in :ref:`analyse-scipy`.
        """
        return scipy.stats.multivariate_normal(self.mean, self.cov)

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

    def to_mcerp(self):
        """Convert to `mcerp`_ objects.

        TODO: document
        """
        # TODO: implement

    def to_soerp(self):
        """Convert to `soerp`_ objects.

        TODO: document
        """
        # TODO: implement

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
        return np.linalg.eigh(self.cov)

    @property
    def n(self):
        """Number of dimensions of the distribution (int).

        Given by the number of parameters.
        """
        return self._n

    @property
    def names(self):
        """Parameter names (`list` of `str`)."""
        return self._names

    @property
    def mean(self):
        """Mean vector (`numpy.ndarray`)."""
        return self._mean

    @property
    def _mean_weighted(self):
        return np.dot(self.precision, self.mean)

    @property
    def cov(self):
        """Covariance matrix (`numpy.ndarray`)."""
        return self._cov

    @property
    def err(self):
        r"""Error vector (`numpy.ndarray`).

        Defined as :math:`\sigma_i = \sqrt{\Sigma_{ii}}`.
        """
        return np.sqrt(np.diag(self.cov))

    @property
    def correlation(self):
        r"""Correlation matrix (`numpy.ndarray`).

        Correlation :math:`C` is related to covariance :math:`\Sigma` via:

        .. math::
            C_{ij} = \frac{ \Sigma_{ij} }{ \sqrt{\Sigma_{ii} \Sigma_{jj}} }
        """
        err = self.err
        return self.cov / np.outer(err, err)

    @property
    def precision(self):
        """Precision matrix (`numpy.ndarray`).

        The inverse of the covariance matrix.

        Sometimes called the "information matrix" or "Hesse matrix".
        """
        return scipy.linalg.pinvh(self.cov)

    def sigma_distance(self, point):
        """Number of standard deviations from the mean (float).

        Also called the Mahalanobis distance.
        See :ref:`theory_sigmas`.
        """
        point = np.asanyarray(point)
        d = self._mean - point
        sigma = np.dot(np.dot(d.T, self.precision), d)
        return np.sqrt(sigma)

    def conditional(self, pars):
        """Conditional `MultiNormal` distribution.

        See :ref:`theory_conditional`.

        TODO: document.
        """
        idx = self._pars_to_idx(pars)
        # TODO: implement

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
        idx = self._pars_to_idx(pars)
        mean = self.mean[idx]
        cov = self.cov[np.ix_(idx, idx)]
        names = [self.names[_] for _ in idx]
        return self.__class__(mean, cov, names)

    def _pars_to_idx(self, pars):
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


def _get_n_from_inputs(mean, cov, names):
    if mean is not None:
        return len(mean)

    if names is not None:
        return len(names)

    if cov is not None:
        return len(cov)

    raise ValueError("Could not determine number of parameters n.")


def sigma_to_containment(sigma, n):
    pass
