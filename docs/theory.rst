.. include:: references.txt

.. _theory:

Theory
======

In this section, we give a bit of theory background concerning the methods
used in ``multinorm``. We give the formulae used, and a reference to where
the formula and a derivation and discussion can be found.

.. note::

    The multivariate normal distribution has very nice mathematical
    properties. Every derived quantity follows either again a multivariate
    normal distribution or a chi-squared distribution.

.. _theory_marginal:

Marginal distribution
---------------------

The marginal distribution can be obtained with the :meth:`~MultiNorm.marginal` method.

You can think of the `marginal distribution`_ as the distribution obtained
by simply ignoring some of the parameters, or by "projecting" the :math:`N`-dimensional
distribution onto the lower-dimensional subspace of parameters of interest.

The marginal distribution of the multivariate normal is again a multivariate normal distribution.

It can be obtained simply by keeping only the parameters of interest in the mean vector
and covariance matrix (drop the parameters that are being marginalised out).

See `here <https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Marginal_distributions>`__.

.. _theory_conditional:

Conditional distribution
------------------------

The conditional distribution can be obtained with the :meth:`~MultiNorm.conditional` method.

The `conditional distribution`_ is given by the "slice" in the :math:`N`-dimensional
distribution when fixing some of the parameters.

The conditional distribution of the multivariate normal is again a multivariate normal distribution.

It can be obtained by partitioning the mean :math:`\mu` and covariance :math:`\Sigma` of the
:math:`N`-dimensional distribution into two part, corresponding to the parameters that are
fixed, and that are kept free.

The formulae to obtain the mean and covariance of the conditional distribution are given
`here <https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions>`_.

.. _theory_sigmas:

Sigmas
------

For the one-dimensional normal distribution :math:`N(\mu, \sigma)` the
probability content within :math:`n * \sigma` is given by roughly
68% for :math:`n=1`, 95% for :math:`n=2` and 99.7% for :math:`n=3`.

What's the equivalent for the :math:`N`-dimensional normal distribution?

For a given mean :math:`\mu` and covariance :math:`\Sigma` and point :math:`p`
one can define a distance :math:`d` via

.. math::

    d = \sqrt{(p - \mu)^T \Sigma^{-1} (p - \mu)}.

The set of equal-distance points is an ellipsoidal surface and has the property
that all points on it have equal probability density. It is the equivalent of the
distance :math:`d = (p - \mu) / \sigma`, i.e. the number of standard deviations
:math:`d = n * \sigma` from the mean.

However, the probability content for a given :math:`n * \sigma` is lower
for the :math:`N`-dimensional distribution. It turns out that :math:`d^2`
has a :math:`\chi^2` distribution with :math:`N` degrees of freedom:

.. math::
    P(d^2) = \chi^2(d^2, N)

That means you can compute the probability content using ``scipy.stats.chi2`` like this:

    >>> import numpy as np
    >>> from scipy.stats import chi2
    >>> n_sigma = np.array([1, 2, 3])
    >>> chi2.cdf(n_sigma ** 2, 1)
    array([0.68268949, 0.95449974, 0.9973002 ])
    >>> chi2.cdf(n_sigma ** 2, 2)
    array([0.39346934, 0.86466472, 0.988891  ])
    >>> chi2.cdf(n_sigma ** 2, 3)
    array([0.19874804, 0.73853587, 0.97070911])

Note that the 1 sigma ellipse in 2D has probability content 39%,
in 3D it's only 20%, and it gets smaller and smaller for higher dimensions.

Also see https://stats.stackexchange.com/questions/331283

For further information see the `Wikipedia Mahalanobis distance`_ page.

The `MultiNorm.to_matplotlib_ellipse` takes an ``n_sigma`` option,
and will return an ellipse that matches the points with Mahalanobis distance
:math:`d^2 = n * \sigma`.

See also `sigma in the corner.py docs`_.

.. _theory_combine:

Combine
-------

The approximation we will use can be found in many textbooks,
e.g. Section 5.6.1 `stats book`_. given :math:`n` Gaussian likelihood
estimates with parameter estimates :math:`x_i` and known parameter errors :math:`\sigma_i`:

.. math::

    p(\mu | {x_i}, {\sigma_i}),

if we define "weights" as inverse square of errors

.. math::

    w_i = 1 / \sigma_i^2,
    \sigma_i = 1 / \sqrt{w_i},

then the joint maximum likelihood estimate error is given by (Equation 5.50):

.. math::

    \mu_0 = \frac{\sum{w_i x_i}}{\sum{w_i}}

and the joint measurement parameter error is given by

.. math::

    w = \sum{w_i}.
