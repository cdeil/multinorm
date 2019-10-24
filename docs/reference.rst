.. include:: references.txt

.. _ref:

References
==========

.. _def:

Definitions
-----------

In the ``multinorm`` package, we use the following variable names:

- ``MultiNorm`` - the multivariate normal (a.k.a. Gaussian) distribution
- ``n`` - the number of dimensions, i.e. number of parameters. Math: :math:`n`
- ``mean`` - the vector of mean values of the distribution. Math: :math:`\mu`
- ``cov`` - covariance matrix of the distribution. Math: :math:`\Sigma`
- ``precision`` -  precision matrix. Math: :math:`\Sigma^{-1}`

Documents
---------

Some useful references for multivariate normal distributions:

- `Wikipedia multivariate normal`_
- `Wikipedia Mahalanobis distance`_

Codes
-----

Other codes related to multivariate normal distributions:

- `Wolfram MultinormalDistribution`_
- `numpy.random.Generator.multivariate_normal`
- `scipy.stats.multivariate_normal`
- `sklearn.covariance`_
- `uncertainties`_
- `statsmodels`_
