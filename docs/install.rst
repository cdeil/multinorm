.. include:: references.txt

Install
=======

To install ``multinorm`` use pip::

    pip install multinorm

This will install the required dependencies if you don't have them already:

- `numpy`_
- `scipy`_
- `pandas`_

There are some built-in methods for plotting using `matplotlib`_.
That optionally dependency has to be installed separately,
`pip install multinorm` will not install matplotlib.

This package consists of a single Python file `multinorm.py`_.
Most users will not care about this implementation detail,
but if you'd like to copy and "vendor" it for some reason,
you can bundle a copy and avoid the extra dependency for just one file and class.
