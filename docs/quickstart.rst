.. include:: references.txt

.. _gs:

Getting started
===============

.. note::

    For a quick and hands-on introduction, start with the
    `multinorm.ipynb`_ Jupyter notebook tutorial,
    then continue reading here.

Import
------

The ``multinorm`` package offers a single class `MultiNorm`,
so you always start like this::

    from multinorm import MultiNorm

.. _gs_create:

Create
------

To create a `MultiNorm` object, pass a ``mean`` vector,
a ``covariance`` matrix (both as Numpy arrays)
and optionally a list of parameter ``names``::

    from multinorm import MultiNorm
    mean = [10, 20, 30]
    covariance = [[1, 0, 0], [0, 4, 0], [0, 0, 9]]
    names = ["a", "b", "c"]
    mn = MultiNorm(mean, covariance, names)

Sometimes the mean and covariance are given directly, e.g. in a publication,
and you would define them in Python code as shown here, or read them from a file.

However, often you obtain these values as the result of a fit of a parametrised
model to data, or estimate them in some other way.

Further examples to create `MultiNorm` objects are here: :ref:`create`

Read only
---------

`MultiNorm` objects should be used read-only objects!

If you need to change something (``mean``, ``covariance``, ``names``), make a new object!

TODO: make read-only as much as possible, the document remaining caveats!

.. _gs_analyse:

Analyse
-------

Once you have a `MultiNorm` object representing a multivariate normal distribution,
you can access the following properties and methods to analyse it.

The object repr only shows the number of dimensions (number of parameters)
``n`` of the distribution:

    >>> mn
    MultiNorm(n=3)

To see the contents, print the object:

    >>> print(mn)
    MultiNorm(n=3)
    names: ['a', 'b', 'c']
    mean: [10. 20. 30.]
    error: [1. 2. 3.]
    cov:
    [[1. 0. 0.]
     [0. 4. 0.]
     [0. 0. 9.]]

You can access the attributes like this:

    >>> mn.n
    3
    >>> mn.mean
    array([10., 20., 30.])
    >>> mn.cov
    array([[1., 0., 0.],
           [0., 4., 0.],
           [0., 0., 9.]])
    >>> mn.names
    ['a', 'b', 'c']

The ``mean`` and ``covar`` are `numpy.ndarray` objects. To be as accurate as possible,
we always cast to 64-bit float on `MultiNorm` initialisation and do all computations
with 64-bit floating point precision, even if 32-bit float or integer numbers are passed in.

    >>> type(mn.mean)
    numpy.ndarray
    >>> mn.mean.dtype
    dtype('float64')

The ``mean`` is a 1-dimensional array, and ``cov`` is a 2-dimensional array:

    >>> mn.mean.shape
    (3,)
    >>> mn.cov.shape
    (3, 3)

Parameter error vector `MultiNorm.error`::

    >>> mn.error
    array([1., 2., 3.])

Precision matrix (the inverse covariance) `MultiNorm.precision`:

    >>> mn.precision
    array([[1.        , 0.        , 0.        ],
           [0.        , 0.25      , 0.        ],
           [0.        , 0.        , 0.11111111]])

Correlation matrix `MultiNorm.correlation`:

    >>> mn.correlation
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])

These are just the basic attributes and properties.

We continue with this example on the :ref:`analyse` page and show
how to really do some analysis with `MultiNorm` objects and methods.

.. _gs_plot:

Plot
----

Plot ellipse using `MultiNorm.to_matplotlib_ellipse`::

    import matplotlib.pyplot as plt
    mn2 = mn.marginal()
    mn2.plot()

Further examples to plot `MultiNorm` objects are here: :ref:`plot`

What next?
----------

The :ref:`create`, :ref:`analyse` and :ref:`plot` tutorial pages contain
further examples. The :ref:`theory` and :ref:`ref` pages contain background
information and definitions, as well as links to other documents and codes.

The full API documentation is here: `MultiNorm`.
Note that you can click on "source" on the right for any method or property,
and read the implementation to see what exactly it does.
It's usually a few lines of straightforward code using Python and Numpy,
so reading the source is recommended.
