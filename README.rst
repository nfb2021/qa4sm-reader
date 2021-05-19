============
qa4sm_reader
============

.. image:: https://github.com/TUW-GEO/qa4sm-reader/workflows/tests/badge.svg
    :target: https://github.com/TUW-GEO/qa4sm-reader/actions?query=workflow%3Atests

.. image:: https://coveralls.io/repos/github/awst-austria/qa4sm-reader/badge.svg?branch=master
    :target: https://coveralls.io/github/awst-austria/qa4sm-reader?branch=master

qa4sm_reader is a python package to read and plot the result files of the `qa4sm service`_.


Installation
============

This package should be installable through pip (not yet tough, see development):

.. code::

    pip install qa4sm_reader

Usage
=====

This package is used to analyze a qa4sm netCDF output file and produce all relevant plots and maps.

Development Setup
=================

The project was setup using `pyscaffold`_ and closely follows the recommendations.

Install Dependencies
--------------------

For Development we recommend creating a ``conda`` environment.

.. code::

    cd qa4sm-reader
    conda env create #  create environment from requirements.rst
    conda activate qa4sm-reader
    python setup.py develop #  Links the code to the environment

To remove the environment again, run:

.. code::

    conda deactivate
    conda env remove -n qa4sm_reader

Testing
-------

For testing, we use ``py.test``:

.. code::

    pytest


The dependencies are automatically installed by `pytest-runner`_ when you run the tests. The test-dependencies are listed in the ``testing`` field inside the ``[options.extras_require]`` section of ``setup.cfg``.
For some reasons, the dependencies are not installed as expected. To workaround, do:

.. code::

    pip install pytest-cov

The files used for testing are included in this package. They are however subject to other `terms and conditions`_.

Known Issues
------------

No known issues - please `open an issue`_ in case you come across a malfunctioning in the package.


.. _qa4sm service: https://qa4sm.eu
.. _pyscaffold: https://pyscaffold.org
.. _pytest-runner: https://github.com/pytest-dev/pytest-runner
.. _terms and conditions: https://qa4sm.eu/terms
.. _open an issue: https://github.com/awst-austria/qa4sm-reader/issues
