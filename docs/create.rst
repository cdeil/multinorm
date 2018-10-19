.. include:: references.txt

.. _create:

Create
======

As we saw in :ref:`gs`, to create a `MultiNorm` object, you
pass a ``mean`` vector, a ``covariance`` matrix (both as Numpy arrays)
and optionally a list of parameter ``names``::

    from multinorm import MultiNorm
    mean = [2, 3]
    covariance = [[1, 2], [3, 4]]
    names = ["a", "b"]
    multi_norm = MultiNorm(mean, covariance, names)

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

.. _create_from_points:

From points
-----------

TODO: show example using emcee where points come from a trace
and ``np.std`` and ``np.cov`` is used to get the inputs

.. _create_from_pub:

From publication
----------------

TODO: show example how to take covar (or par errors) from a
publication or blog post, i.e. as inputs.
