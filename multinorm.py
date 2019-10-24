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
import numpy as np
from scipy.linalg import eigh, block_diag
from scipy.stats import multivariate_normal

__all__ = ["MultiNorm"]

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass


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

    def __str__(self):
        return (
            f"{self.__class__.__name__} with n={self.n} parameters:\n"
            f"{self.parameters!s}"
        )

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
        diagonal covariance matrix with :math:`\Sigma_{ii} = \sigma_i^2`.

        For a given `correlation`, or in general, this will create
        a `MultiNorm` with a covariance matrix such that it's
        `err` and `correlation` match the one specified here
        (up to rounding errors).

        See :ref:`create_from_fit` and :ref:`create_from_pub`.

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

        Returns
        -------
        `MultiNorm`
        """
        if err is None:
            raise ValueError("Must set err parameter")

        err = np.asarray(err, dtype=float)
        n = len(err)

        if correlation is None:
            correlation = np.eye(n)
        else:
            correlation = np.asarray(correlation, dtype=float)

        cov = correlation * np.outer(err, err)

        return cls(mean, cov, names)

    @classmethod
    def from_samples(cls, samples, names=None):
        """Create `MultiNorm` from parameter samples.

        Usually the samples are from some distribution
        and creating this `MultiNorm` distribution is an
        estimate / approximation of that distribution of interest.

        See :ref:`create_from_samples`.

        Parameters
        ----------
        samples : numpy.ndarray
            Array of data points with shape ``(n_samples, n_par)``.
        names : list
            Parameter names

        Returns
        -------
        `MultiNorm`
        """
        mean = np.mean(samples, axis=0)
        cov = np.cov(samples, rowvar=False)
        return cls(mean, cov, names)

    @classmethod
    def from_stack(cls, distributions):
        """Create `MultiNorm` by stacking distributions.

        Stacking means the ``names`` and ``mean`` vectors
        are concatenated, and the ``cov`` matrices are
        combined into a block diagonal matrix, with zeros
        for the off-diagonal parts.

        This represents the combined measurement, assuming
        the individual distributions are for different parameters.

        See :ref:`create_from_stack` and :ref:`theory_stack`.

        Parameters
        ----------
        distributions : list
            Python list of `MultiNorm` distributions.

        Returns
        -------
        `MultiNorm`
        """
        names = np.concatenate([_.names for _ in distributions])
        cov = block_diag(*[_.cov for _ in distributions])
        mean = np.concatenate([_.mean for _ in distributions])
        return cls(mean, cov, names)

    @classmethod
    def from_product(cls, distributions):
        """Create `MultiNorm` product distribution.

        This represents the joint likelihood distribution, assuming
        the individual distributions are from independent measurements.

        See :ref:`create_from_product` and :ref:`theory_product`.

        Parameters
        ----------
        distributions : list
            Python list of `MultiNorm` distributions.

        Returns
        -------
        `MultiNorm`
        """
        names = distributions[0].names

        precisions = [_.precision for _ in distributions]
        precision = np.sum(precisions, axis=0)
        cov = _matrix_inverse(precision)

        means_weighted = [_.precision @ _.mean for _ in distributions]
        means_weighted = np.sum(means_weighted, axis=0)
        mean = cov @ means_weighted
        return cls(mean, cov, names)

    @classmethod
    def make_example(cls, n_par=3, n_fix=0, random_state=0):
        """Create `MultiNorm` example for testing.

        This is a factory method that allows the quick creation
        of example `MultiNorm` with any number of parameters for testing.

        See: :ref:`create_make_example`.

        Parameters
        ----------
        n_par : int
            Number of parameters
        n_fix : int
            Number of fixed parameters in addition to ``n_par``.
        random_state :
            Seed (int) - default: 0
            Put ``None`` to choose random seed.
            Can also pass `numpy.random.mtrand.RandomState` object.

        Returns
        -------
        `MultiNorm`
        """
        n = n_par + n_fix
        rng = np.random.RandomState(random_state)
        mean = rng.normal(size=n)

        s = rng.normal(size=(n_par, n_par))
        cov1 = s @ s.T

        cov2 = np.zeros((n, n))
        cov2[:n_par, :n_par] = cov1

        return cls(mean, cov2)

    @property
    def scipy(self):
        """Scipy representation (`scipy.stats.multivariate_normal`).

        Used for many computations internally.
        """
        return self._scipy

    @property
    def parameters(self):
        """Parameter table (`pandas.DataFrame`).

        Index is "name", columns are "mean" and "err"
        """
        import pandas as pd
        data = {"mean": self.mean, "err": self.err}
        index = pd.Index(self.names, name="name")
        return pd.DataFrame(data, index)

    def confidence_interval(self, n_sigma=1):
        """Confidence interval table (`pandas.DataFrame`).

        Index is "name", columns are "lo" and "hi"
        """
        import pandas as pd
        d = n_sigma * self.err
        data = {"lo": self.mean - d, "hi": self.mean + d}
        index = pd.Index(self.names, name="name")
        return pd.DataFrame(data, index)

    def to_uncertainties(self):
        """Convert to `uncertainties`_ objects.

        The `uncertainties`_ package makes it easy to
        do error propagation on derived quantities.

        See :ref:`analyse-error`.

        Returns
        -------
        tuple (length ``n``) of ``uncertainties.core.AffineScalarFunc``
        """
        from uncertainties import correlated_values

        return correlated_values(self.scipy.mean, self.scipy.cov, self.names)

    def to_xarray(self, fcn="pdf", n_sigma=3, num=100):
        """Make image of the distribution (`xarray.DataArray`).

        This is mostly useful for visualisation, not used by other methods.

        Parameters
        ----------
        fcn : str
            Function to compute data values. Choices:

            - "pdf" (`pdf`)
            - "logpdf" (`logpdf`)
            - "stat" (``-2 * logpdf``)
            - "sigma" (`sigma_distance`)

        n_sigma : int
            Number of standard deviations. Controls image coordinate range.
        num : int
            Number of pixels in each dimension. Controls image resolution.

        Returns
        -------
        `xarray.DataArray`
        """
        from xarray import DataArray

        coords = [
            np.linspace(row["lo"], row["hi"], num)
            for _, row in self.confidence_interval(n_sigma).iterrows()
        ]
        points = [_.flatten() for _ in np.meshgrid(*coords)]
        points = np.array(points).T

        if fcn == "pdf":
            data = self.pdf(points)
        elif fcn == "logpdf":
            data = self.logpdf(points)
        elif fcn == "stat":
            data = -2 * self.logpdf(points)
        elif fcn == "sigma":
            data = self.sigma_distance(points)
        else:
            raise ValueError(f"Invalid fcn: {fcn!r}")

        data = data.reshape(self.n * (num,))

        return DataArray(data, coords, self.names)

    def error_ellipse(self, n_sigma=1):
        """Error ellipse parameters.

        TODO: document formulae and give example in the docs.

        Parameters
        ----------
        n_sigma : int
            Number of standard deviations. See :ref:`theory_sigmas`.

        Returns
        -------
        dict
            Keys "xy" (center, tuple), and floats  "width", "height", "angle"
        """
        # See https://stackoverflow.com/questions/12301071
        xy = self.scipy.mean
        vals, vecs = eigh(self.scipy.cov)
        width, height = 2 * n_sigma * np.sqrt(vals)
        angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        return {"xy": xy, "width": width, "height": height, "angle": angle}

    def to_matplotlib_ellipse(self, n_sigma=1, **kwargs):
        """Create error ellipse (`matplotlib.patches.Ellipse`).

        See :ref:`plot`.

        Parameters
        ----------
        n_sigma : int
            Number of standard deviations. See :ref:`theory_sigmas`.

        Returns
        -------
        `matplotlib.patches.Ellipse`
        """
        if self.n != 2:
            raise ValueError(
                "Ellipse only available for n=2. "
                "To select parameters, call ``marginal`` or ``conditional`` first."
            )

        from matplotlib.patches import Ellipse

        ellipse = self.error_ellipse(n_sigma)

        return Ellipse(**ellipse, **kwargs)

    def plot(self, ax=None, n_sigma=1, **kwargs):
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax
        ellipse = self.to_matplotlib_ellipse(n_sigma, **kwargs)
        ax.add_artist(ellipse)

    @property
    def n(self):
        """Number of dimensions (`int`)."""
        return self.scipy.dim

    @property
    def mean(self):
        """Parameter mean values (`numpy.ndarray`)."""
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
        r"""Parameter errors (`numpy.ndarray`).

        Defined as :math:`\sigma_i = \sqrt{\Sigma_{ii}}`.
        """
        return np.sqrt(np.diag(self.scipy.cov))

    @property
    def correlation(self):
        r"""Correlation matrix (`numpy.ndarray`).

        Correlation :math:`C` is related to covariance :math:`\Sigma` via:

        .. math::
            C_{ij} = \frac{ \Sigma_{ij} }{ \sqrt{\Sigma_{ii} \Sigma_{jj}} }
        """
        return self.cov / np.outer(self.err, self.err)

    @property
    def precision(self):
        """Precision matrix (`numpy.ndarray`).

        The inverse of the covariance matrix.

        Sometimes called the "information matrix" or "Hesse matrix".
        """
        return _matrix_inverse(self.scipy.cov)

    def drop(self, pars):
        """Drop parameters.

        This simply removes the entry from the `mean` vector,
        and the corresponding column and row from the `cov` matrix.

        The computation is the same as `MultiNorm.marginal`,
        only here the parameters to drop are given, and there
        the parameters to keep are given.

        Parameters
        ----------
        pars : list
            Parameters to fix (indices or names)

        Returns
        -------
        `MultiNorm`
        """
        mask = np.invert(self._name_index.get_mask(pars))
        return self._subset(mask)

    def marginal(self, pars):
        """Marginal distribution.

        See :ref:`theory_marginal`.

        Parameters
        ----------
        pars : list
            List of parameters (integer indices)

        Returns
        -------
        `MultiNorm`
        """
        mask = self._name_index.get_mask(pars)
        return self._subset(mask)

    def _subset(self, mask):
        mean = self.scipy.mean[mask]
        cov = self.scipy.cov[np.ix_(mask, mask)]
        names = self._name_index.get_names(mask)
        return self.__class__(mean, cov, names)

    def conditional(self, pars, values=None):
        """Conditional `MultiNorm` distribution.

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
        `MultiNorm`
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

        mean = mean1 + cov12 @ np.linalg.solve(cov22, values - mean2)
        cov = cov11 - cov12 @ np.linalg.solve(cov22, cov21)

        return self.__class__(mean, cov, names)

    def fix(self, pars):
        """Fix parameters.

        See :ref:`theory_fix`.

        Parameters
        ----------
        pars : list
            Parameters to fix (indices or names)

        Returns
        -------
        `MultiNorm`
        """
        # mask of parameters to keep (that are not fixed)
        mask = np.invert(self._name_index.get_mask(pars))
        names = self._name_index.get_names(mask)

        mean = self.scipy.mean[mask]
        precision = self.precision[np.ix_(mask, mask)]
        cov = _matrix_inverse(precision)
        return self.__class__(mean, cov, names)

    def sigma_distance(self, points):
        """Number of standard deviations from the mean.

        Also called the Mahalanobis distance.
        See :ref:`theory_sigmas`.

        Parameters
        ----------
        points : numpy.ndarray
            Point coordinates, 2-dim, shape ``(n_points, n_par)``.

        Returns
        -------
        `numpy.ndarray`
            1-dim, shape ``(n_points,)``
        """
        # https://stackoverflow.com/questions/27686240/calculate-mahalanobis-distance-using-numpy-only
        points = np.atleast_2d(points)
        d = self.mean - points
        d2 = np.einsum('nj,jk,nk->n', d, self.precision, d)
        return np.sqrt(np.squeeze(d2))

    def pdf(self, points):
        """Probability density function.

        Calls ``pdf`` method of `scipy.stats.multivariate_normal`.

        Parameters
        ----------
        points : numpy.ndarray
            Point coordinates, 2-dim, shape ``(n_points, n_par)``.

        Returns
        -------
        `numpy.ndarray`
            1-dim, shape ``(n_points,)``
        """
        return self.scipy.pdf(points)

    def logpdf(self, points):
        """Natural log of PDF.

        Calls ``logpdf`` method of `scipy.stats.multivariate_normal`.

        Parameters
        ----------
        points : numpy.ndarray
            Point coordinates, 2-dim, shape ``(n_points, n_par)``.

        Returns
        -------
        `numpy.ndarray`
            1-dim, shape ``(n_points,)``
        """
        return self.scipy.logpdf(points)

    def sample(self, size=1, random_state=None):
        """Draw random samples.

        Calls ``rvs`` methods of `scipy.stats.multivariate_normal`

        Parameters
        ----------
        size : int
            Numpy of samples to draw
        random_state :
            Seed (int) - default: 0
            Put ``None`` to choose random seed.
            Can also pass `numpy.random.mtrand.RandomState` object.

        Returns
        -------
        points : numpy.ndarray
            Point coordinates, 2-dim, shape ``(n_points, n_par)``.
        """
        return np.atleast_2d(self.scipy.rvs(size, random_state))


class _NameIndex:
    """Parameter name index."""

    def __init__(self, names, n):
        if names is None:
            names = [f"par_{idx}" for idx in range(n)]
        else:
            if len(names) != n:
                raise ValueError(f"len(names) = {len(names)}, expected n={n}")

        # TODO: change to Numpy array!
        self.names = list(names)

    def get_idx(self, pars):
        """Create parameter index array.

        Supports scalar, list and array input pars
        Support parameter indices (int) and names (str)
        """
        # TODO: should we support scalar?
        # TODO: support `np.int32` also
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
        mask = np.zeros(len(self.names), dtype=bool)
        mask[idx] = True
        return mask

    def get_names(self, mask):
        # This works for an index array or mask for the selection
        return list(np.array(self.names)[mask])
