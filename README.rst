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
# TODO

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

    python setup.py pytest


The dependencies are automatically installed by `pytest-runner`_ when you run the tests. The test-dependencies are listed in the ``testing`` field inside the ``[options.extras_require]`` section of ``setup.cfg``.
For some reason, the dependencies are not installed as expected. To workaround, do:

.. code::

    pip install pytest-cov

The files used for testing are included in this package. They are however subject to other `terms and conditions`_.

Known Issues
------------

When creating a boxplot with five or more boxes, the text might overlap.
(see tests/test_ncplot/test_boxplot_GLDAS_nan_default() and tests/test_ncplot/test_boxplot_ISMN_nan_default() )

For some


.. _qa4sm service: https://qa4sm.eodc.eu
.. _pyscaffold: https://pyscaffold.org
.. _pytest-runner: https://github.com/pytest-dev/pytest-runner
.. _terms and conditions: https://qa4sm.eodc.eu/terms
