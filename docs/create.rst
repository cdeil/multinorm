.. include:: references.txt

.. _create:

Create
======

As we saw in :ref:`gs`, to create a `MultiNorm` object, you
pass a ``mean`` vector, a ``covariance`` matrix (both as Numpy arrays)
and optionally a list of parameter ``names``::

    from multinorm import MultiNorm
    mean = [10, 20, 30]
    covariance = [[1, 0, 0], [0, 4, 0], [0, 0, 9]]
    names = ["a", "b", "c"]
    mn = MultiNorm(mean, covariance, names)

But where do these things come from?

On this page, we look at the most common scenarios.

.. _create_from_fit:

From fit
--------

TODO: show example using scipy.optimise.curve_fit`

To use ``multinorm``, we first need to fit some parameterised model
to obtain a best-fit parameter vector and covariance matrix.

Let's use `scipy.optimize_curve_fit`_ to fit some data.

TODO: show example using iminuit

http://www.statsmodels.org/devel/examples/notebooks/generated/chi2_fitting.html
https://github.com/cdeil/pyfit/blob/master/fitting_tutorial/src/tests/chi2_example.py
https://lmfit.github.io
https://iminuit.readthedocs.io
https://sherpa.readthedocs.io

.. _create_from_samples:

From samples
------------

A common way to analyse likelihood or in Bayesian analyses the
posterior probability distributions is to use MCMC methods that
sample the distribution. E.g. `emcee`_ or `pymc`_ are Python packages
that generate this kind of output.

Estimating the multivariate normal distribution from samples well
can be difficult, there are many methods with different trade-offs.
We recommend using a different package for this task, e.g. `sklearn.covariance`_.

That said, there is a method :meth:`MultiNorm.from_samples` that calls :func:`numpy.std`
to compute the mean vector, and :func:`numpy.cov` to compute what's sometimes called
the "empirical" multivariate normal estimate.

Samples should always be given as 2-dimensional arrays with shape ``(n_dim, n_samples)``.

::

    >>> samples = mn.sample(size=100, random_state=0)
    >>> MultiNorm.from_samples(samples, names=mn.names)
    MultiNorm with n=3 parameters:
               mean       err
    name
    a      9.875816  0.980901
    b     20.212505  1.973948
    c     30.301562  3.093609

.. _create_from_pub:

From publication
----------------

TODO: show example how to take covar (or par errors) from a
publication or blog post, i.e. as inputs.

.. _create_from_product:

From product
------------

TODO: document :meth:`MultiNorm.from_product`

.. _create_make_example:

Make example
------------

TODO: document :meth:`MultiNorm.make_example`
