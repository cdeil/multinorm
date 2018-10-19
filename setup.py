from setuptools import setup

long_description = """
multinorm - Multivariate Normal Distributions for Humans.

A Python class to work with model fit results
(parameters and the covariance matrix).

- Code: https://github.com/cdeil/multinorm
- Docs: https://multinorm.readthedocs.io
- License: BSD-3-Clause
"""

setup(
    name="multinorm",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    install_requires=["numpy", "scipy"],
    description="Multivariate Normal Distributions for Humans",
    long_description=long_description,
    author="Christoph Deil",
    author_email="Deil.Christoph@gmail.com",
    url="https://github.com/cdeil/multinorm/",
    license="BSD",
    py_modules=["multinorm"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
    ],
)
