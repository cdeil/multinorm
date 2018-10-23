.. include:: references.txt

.. _analyse:

Analyse
=======

Example
-------

A basic example and properties of `MultiNorm` were shown in :ref:`gs_analyse`.

On this page we continue with analysis methods using same example::

    from multinorm import MultiNorm
    mean = [10, 20, 30]
    covariance = [[1, 0, 0], [0, 4, 0], [0, 0, 9]]
    names = ["a", "b", "c"]
    mn = MultiNorm(mean, covariance, names)

Marginal
--------

TODO: marginal

.. _analyse-conditional:

Conditional
-----------

TODO: conditional

.. _analyse-scipy:

Scipy
-----

The `scipy.stats.multivariate_normal`_ class is similar to `MultiNorm`,
it contains a mean vector and covariance matrix. However, at this time,
there is no overlap in functionality.

Feedback on the design of `MultiNorm` is very welcome! E.g. we could also make
`MultiNorm` a "frozen distribution", i.e. read-only, with cached properties. The internal
data member could be a `scipy.stats.multivariate_normal`_ object directly,
and we could re-expose all functionality diretly. Probably sub-classing isn't
a good idea, because we'd give up full control of the API?

If you want to use one of the  `scipy.stats.multivariate_normal`_ methods
and have a `MultiNorm` object, first convert it via the :meth:`~MultiNorm.to_scipy` method::

    >>> s = mn.to_scipy()
    >>> type(s)
    scipy.stats._multivariate.multivariate_normal_frozen


Draw random variate samples from the distribution::

    >>> points = s.rvs(size=2, random_state=0)
    >>> points
    array([[10.97873798, 20.80031442, 35.29215704],
           [ 9.02272212, 23.73511598, 36.7226796 ]])

Points are always given as arrays with shape ``(n_dim, n_points)``.

Evaluate the probability density function (PDF)::

    >>> s.pdf(points)
    array([1.27661616e-03, 9.31966590e-05])

Or the ``log(pdf)`` (natural logarithm)::

    >>> s.logpdf(points)
    array([-6.66354232, -9.28079868])

There is also a ``cdf`` and ``logcdf`` method for the cumulative distribution function,
as well as ``entropy``, and a ``cov_info`` which is undocumented, but seems to contain
some covariance matrix decomposition.

Error propagation
-----------------

TODO: to_uncertainties, to_soerp, to_mcerp

Joint
-----

TODO: from_product

Sigmas
------

TODO: sigma_distance
