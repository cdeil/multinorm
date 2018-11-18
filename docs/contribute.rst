.. include:: references.txt

.. _contribute:

Contribute
==========

This package is very new, there hasn't been any user feedback or review yet.
Very likely the API and implementation can be improved.

Please give feedback to help make it better!

Github
------

Contributions to `multinorm` are welcome any time
on Github: https://github.com/cdeil/multinorm

- If you find an issue, please file a bug report.
- If you're missing a feature, please file a request.
- If you have the skills and time, please send a pull request.

Pull requests should be small and easy to review.
If the work takes more than an hour, please open an issue
describing what you plan to do first to get some feedback.

Develop
-------

To work on ``multinorm``, first get the latest version::

    git clone https://github.com/cdeil/multinorm.git
    cd multinorm

Everything is where you'd expect it, i.e. the files to edit are:

- Code: `multinorm.py`_
- Tests: `test_multinorm.py`_
- Docs: RST files in `docs`_

Install
-------

To hack on ``multinorm``, you need to have a development environment
with all packages and tools installed.

I'm using ``conda``, there it's easy to create the environment::

    conda env create -f environment.yml
    conda activate multinorm

With the virtual environment active, run this command::

    pip install -e .

This installs ``multinorm`` in editable mode, meaning a pointer
is put in your site-packages to the current source folder, so
that after editing the code you only have to re-start python
and re-run to get this new version, and not run an install command again.

Tests
-----

Run all tests::

    pytest -v

Run tests and check coverage::

    pytest -v --cov=multinorm --cov-report=html
    open htmlcov/index.html

Code style
----------

We use the `black`_ code style. To apply it in-place to all files::

    black .

Docs
----

To build the docs::

    cd docs
    make clean && make html
    open _build/html/index.html

Then for any other tasks go back to the top level of the package::

    cd ..

Release
-------

To make a release for this package, follow the following steps

#. check that the tests and docs build are OK
#. check via ``git tag`` or at https://pypi.org/project/multinorm what the next version should be
#. ``git clean -fdx``
#. ``git tag v0.1`` (substitute actual version number here and in the following steps)
#. ``python setup.py build sdist``
#. check the package in ``dist`` (should automate somehow)
#.  ``twine upload dist/*``
#. ``git push --tags``

We should automate this. I didn't have time yet to try them out, but these look interesting:

- https://github.com/pyscaffold/pyscaffold
- https://github.com/regro/rever
- https://github.com/noirbizarre/bumpr
