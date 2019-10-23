.. include:: references.txt

.. _theory:

Theory
======

This section gives a bit of theory background about the multivariate normal
distribution, the goal is to have all formulae used in the `MultiNorm` methods
documented here.

The `Wikipedia normal distribution`_ and the `Wikipedia multivariate normal`_
pages contain most of what we use, so do most textbooks that cover the
multivariate normal distribution.

Here we just give the results, we don't derive them.

.. note::

    The multivariate normal distribution has very nice mathematical
    properties. Most operations are given by linear algebra.
    Every derived quantity follows either again a multivariate
    normal distribution or a chi-squared distribution.

.. _theory_pdf:

Probability density
-------------------

The (univariate) normal distribution :math:`x \sim N(\mu, \sigma^2)`
for a variable :math:`x` with mean :math:`\mu` and variance :math:`\sigma^2`
has the probability density function (``pdf``)

.. math::
    f(x) = \frac{1}{\sqrt{(2 \pi) \sigma^2}}
           \exp\left( -\frac{1}{2} \frac{(x - \mu)^2}{\sigma^2}\right).

The variance :math:`\Sigma` is the square of the standard deviation :math:`\sigma`:

.. math::
    \Sigma = \sigma ^ 2

The multivariate normal distribution :math:`x \sim N(\mu, \Sigma)`
for a variable :math:`x` (vector of length ``N``), with mean :math:`\mu` (vector of length ``N``)
and covariance matrix :math:`\Sigma` (square symmetric matrix of shape ``(N, N)``) has the pdf:

.. math::
    f(x) = \frac{1}{\sqrt{(2 \pi)^N \det \Sigma}}
           \exp\left( -\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu) \right),

Note that the univariate distribution is a special case of the multivariate distribution
for ``N=1``, one just has to write :math:`(x - \mu)^2 / \sigma^2` as
:math:`(x - \mu)^T \Sigma^{-1} (x - \mu)` in the form that works for vectors :math:`x` and :math:`\mu`
and a matrix :math:`\Sigma`.

The inverse of the covariance matrix :math:`\Sigma` is called the precision matrix :math:`Q`:

.. math::
    Q = \Sigma^{-1}

For the one-dimensional distribution the precision is the inverse variance, :math:`Q = 1 / \sigma^2`,
i.e. the precision is large if the variance and standard deviation are small.

.. note::

    Note that for a measurement :math:`\mu \pm \sigma` where :math:`sigma` represents the parameter
    error, the covariance matrix contains the squared errors :math:`\Sigma_{ii} = \sigma_i ^2` on the diagonal,
    and to obtain the error from the covariance matrix one uses :math:`\sigma_i = \sqrt{\Sigma_ii}`.

    Sometimes the covariance matrix is also called "error matrix". We avoid that term because
    it is confusing, given that the matrix doesn't contain errors, but the squared errors
    on the diagonal.

    Sometimes also the term "variance-covariance matrix" is used instead of just "covariance matrix".
    This is accurate: the matrix does contain the parameter variances on the diagonal,
    and the off-diagonal terms are the covariances.

.. _theory_compute:

Compute errors
--------------

If we define the "fit statistic" function as minus two times the log likelihood:

.. math::
    s(x) = - 2 \log(f(x))

then we see that if :math:`f` is a multinormal distribution, then :math:`s` will be a parabola:

.. math::
    s(x) = s_0 + \left(\frac{x - \mu}{\sigma}\right)^2

with the best-fit statistic value :math:`s_0` of

.. math::
    s_0 = TODO

The second derivative (called Hessian) :math:`\frac{d^2s}{dx^2}` is given by:

.. math::
    H = \frac{d^2s}{dx^2}(x)  = 2 / \sigma^2

So for a given "fit statistic" function, the covariance matrix can be computed
via the inverse Hessian:

.. math::

    If you define a fit statistic function as :math:`s(x) = - 2 \log(f(x))`,
    then the Hessian :math:`H = \delta s / \del x \del y` is twir
    \Sigma = 2 H^{-1}

.. note::

    The covariance matrix is

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

.. _theory_fix:

Fix parameters
--------------

This method is used e.g. in MINUIT, see Section 1.3.1 here:
http://lmu.web.psi.ch/docu/manuals/software_manuals/minuit2/mnerror.pdf

As far as I can tell, it gives the same results as conditional (see `test_conditional_vs_fix`).

TODO: work out the math of why that is the case and document it here.

Add note that for MVN the covar matrix for conditional doesn't depend on parameter values.

TODO: document and make example in the analyse section using iminuit.

.. _theory_stack:

Stacked distribution
--------------------

TODO: document :meth:`MultiNorm.from_stack`

.. _theory_product:

Product distribution
--------------------

TODO: improve this section: https://github.com/cdeil/multinorm/issues/13

We should give the full equations, the ones below are the special case for distributions without correlations.

The approximation we will use can be found in many textbooks,
e.g. Section 5.6.1 `stats book`_. given :math:`n` Gaussian likelihood
estimates with parameter estimates :math:`x_i` and known parameter errors :math:`\sigma_i`:

.. math::

    p(\mu | {x_i}, {\sigma_i}),

if we define "weights" as inverse square of errors

.. math::

    w_i = 1 / \sigma_i^2,
    \sigma_i = 1 / \sqrt{w_i},

then the from_product maximum likelihood estimate error is given by (Equation 5.50):

.. math::

    \mu_0 = \frac{\sum{w_i x_i}}{\sum{w_i}}

and the from_product measurement parameter error is given by

.. math::

    w = \sum{w_i}.

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
