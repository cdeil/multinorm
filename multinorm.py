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
    """

    def __init__(self, mean=None, cov=None):
        # let `multivariate_normal` do all input validation
        self._scipy = multivariate_normal(mean, cov, allow_singular=True)

    def __str__(self):
        return (
            f"{self.__class__.__name__} with n={self.n} parameters:\n"
            f"{self.summary_dataframe()!s}"
        )

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented

        return (
            (self.mean == other.mean).all()
            and (self.cov == other.cov).all(axis=None)
        )

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
    def error(self):
        r"""Parameter errors (`numpy.ndarray`).

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
        return self.cov / np.outer(self.error, self.error)

    @property
    def precision(self):
        """Precision matrix (`numpy.ndarray`).

        The inverse of the covariance matrix.

        Sometimes called the "information matrix" or "Hesse matrix".
        """
        return _matrix_inverse(self.cov)

    @property
    def scipy(self):
        """Scipy representation (`scipy.stats.multivariate_normal`).

        Used for many computations internally.
        """
        return self._scipy

    @classmethod
    def from_error(cls, mean=None, error=None, correlation=None):
        r"""Create `MultiNorm` from parameter errors.

        With errors :math:`\sigma_i` this will create a
        diagonal covariance matrix with :math:`\Sigma_{ii} = \sigma_i^2`.

        For a given `correlation`, or in general, this will create
        a `MultiNorm` with a covariance matrix such that it's
        `error` and `correlation` match the one specified here
        (up to rounding errors).

        See :ref:`create_from_fit` and :ref:`create_from_pub`.

        Parameters
        ----------
        mean : numpy.ndarray
            Mean vector
        error : numpy.ndarray
            Error vector
        correlation : numpy.ndarray
            Correlation matrix

        Returns
        -------
        `MultiNorm`
        """
        if error is None:
            raise ValueError("Must set error parameter")

        error = np.asarray(error, dtype=float)
        n = len(error)

        if correlation is None:
            correlation = np.eye(n)
        else:
            correlation = np.asarray(correlation, dtype=float)

        cov = correlation * np.outer(error, error)

        return cls(mean, cov)

    @classmethod
    def from_samples(cls, samples):
        """Create `MultiNorm` from parameter samples.

        Usually the samples are from some distribution
        and creating this `MultiNorm` distribution is an
        estimate / approximation of that distribution of interest.

        See :ref:`create_from_samples`.

        Parameters
        ----------
        samples : numpy.ndarray
            Array of data points with shape ``(n_samples, n_par)``.

        Returns
        -------
        `MultiNorm`
        """
        mean = np.mean(samples, axis=0)
        cov = np.cov(samples, rowvar=False)
        return cls(mean, cov)

    @classmethod
    def from_stack(cls, distributions):
        """Create `MultiNorm` by stacking distributions.

        The ``mean`` vectors are concatenated, and the ``cov`` matrices are
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
        # TODO: input validation to give good error
        cov = block_diag(*[_.cov for _ in distributions])
        mean = np.concatenate([_.mean for _ in distributions])
        return cls(mean, cov)

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
        precisions = [_.precision for _ in distributions]
        precision = np.sum(precisions, axis=0)
        cov = _matrix_inverse(precision)

        means_weighted = [_.precision @ _.mean for _ in distributions]
        means_weighted = np.sum(means_weighted, axis=0)
        mean = cov @ means_weighted
        return cls(mean, cov)

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

    def summary_dataframe(self, n_sigma=None):
        """Summary table (`pandas.DataFrame`).

        - "mean" -- `mean`
        - "error" - `error`
        - ("lo", "hi") - confidence interval (if ``n_sigma`` is set)

        Parameters
        ----------
        n_sigma : float
            Number of standard deviations

        Returns
        -------
        `pandas.DataFrame`
            Summary table with one row per parameter
        """
        import pandas as pd

        df = pd.DataFrame(
            data={"mean": self.mean, "error": self.error},
        )

        if n_sigma is not None:
            df["lo"] = self.mean - n_sigma * self.error
            df["hi"] = self.mean + n_sigma * self.error

        return df

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

        return correlated_values(self.mean, self.cov)

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
            for _, row in self.summary_dataframe(n_sigma).iterrows()
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

        return DataArray(data, coords)

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
        vals, vecs = eigh(self.cov)
        width, height = 2 * n_sigma * np.sqrt(vals)
        angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

        return {"xy": self.mean, "width": width, "height": height, "angle": angle}

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
        mask = self.make_index_mask(pars)
        mean = self.mean[mask]
        cov = self.cov[np.ix_(mask, mask)]
        return self.__class__(mean, cov)

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
        mask2 = self.make_index_mask(pars)
        mask1 = np.invert(mask2)

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

        mean = mean1 + cov12 @ np.linalg.solve(cov22, values - mean2)
        cov = cov11 - cov12 @ np.linalg.solve(cov22, cov21)

        return self.__class__(mean, cov)

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
        mask = np.invert(self.make_index_mask(pars))

        mean = self.mean[mask]
        precision = self.precision[np.ix_(mask, mask)]
        cov = _matrix_inverse(precision)
        return self.__class__(mean, cov)

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
        d2 = np.einsum("nj,jk,nk->n", d, self.precision, d)
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

    def make_index_mask(self, pars):
        """Make index mask.
        TODO: document
        """
        if pars is None:
            return np.ones(self.n, dtype=bool)

        # pars = np.array(pars)
        #
        # if len(pars) != self.n:
        #     raise ValueError()

        # Defer to Numpy for indexing
        mask = np.zeros(self.n, dtype=bool)
        mask[pars] = True

        return mask
