.. _Get_started:

******************************************************************************
Quick installation
******************************************************************************

Requirements
==============================================================================

The following packages are required to install and use CyLag:
*cython >= 0.29*,
*numpy*,
*scipy*,
*mpi4py*,
*h5py*,
*pytest*,
*matplotlib*,
*setuptools*,
*sphinx*, and *sphinx-gallery* (for the documentation).

If you already use openTELEMAC_, all Python requirements should already be installed (except Sphinx).
If not, the recommanded practice is to install a python virtual environment:

.. code-block:: bash
  
  python3 -m venv pyenv/cylag


After creation, activate venv :

.. code-block:: bash

  source pyenv/cylag/bin/activate


Install all requirements :

.. code-block:: bash

  pip install -r REQUIREMENTS.txt


Compile and import CyLag
==============================================================================

CyLag is primarily implemented in Cython_ and Python_, combining the performance of compiled code with the flexibility of a high-level scripting language.
Before using CyLag, the library must be compiled. 
Compilation is performed by executing the ``setup.py`` script located in the root directory of the repository.
Once compiled, CyLag can be imported and used as a standard Python package.

- Add CyLag to the Python path by copying the following line into ~/.bashrc:

.. code-block:: bash

  export PYTHONPATH=$PYTHONPATH:</path/to/cylag>

- Compile CyLag:

.. code-block:: bash

  python3 setup.py build_ext --inplace

- Compile CyLag in debug mode:

.. code-block:: bash

  python3 setup.py build_ext --inplace --debug

- Test importing CyLag:

.. code-block:: bash

  python3
  import cylag

- When recompiling the code, you might want to clean the previously built extensions by using:

.. code-block:: bash

  python3 setup.py clean


Repository
==============================================================================

The CyLag repository is organized into several directories, including:

- ``cylag``: source code of the library;
- ``examples``: validation test cases and benchmark simulations;
- ``tests``: unit tests used to verify implementation and non-regression;
- ``docs``: Sphinx documentation.


Build documentation
==============================================================================

The CyLag repository also includes a dedicated Sphinx-based documentation, providing installation instructions, theoretical background, API references, tutorials, and practical examples to facilitate the use and development of the code.

- Install Sphinx:

.. code-block:: bash

  pip install sphinx sphinx-gallery sphinx-autodoc-typehints


- Build the documentation (html):

.. code-block:: bash

  cd docs
  make html

The documentation should be available in ``docs/build/html/index.html``.

- Build the documentation (latex/pdf):

.. code-block:: bash

  cd docs
  make latexpdf

.. note::

  the gallery of example and API are not included with the latex build of the documentation.


Setup a simulation and run CyLag
==============================================================================

CyLag follows a **component-based architecture** with clear separation of concerns.
To set up a simulation, the user needs to define three main objects:

- ``EulerianFieldSet``: contains the Eulerian fields computed with a 2D or 3D hydrodynamic solver such as TELEMAC,
- ``LagrangianParticleSet``: contains the particles modeled by CyLag and their physical properties,
- ``Solver``: solves the Lagrangian Stochastic Models and updates the particle state vector. It takes ``EulerianFieldSet`` and ``LagrangianParticleSet`` as inputs, as well as a ``parameters`` dictionary containing the main solver options.

CyLag is run using its Python API; to launch a simulation, one has to define a Python script in which cylag is imported and cylag objects are defined. The simulation script usually looks like this:

.. code-block:: python

   import cylag
   fset = cylag.EulerianFieldSet(...)
   pset = cylag.LagrangianParticleSet(...)
   parameters = {...}
   cylag_solver = cylag.Solver(fset, pset, parameters)
   cylag_solver.solve()


Each object has its own inputs and parameters.
See the gallery of ``examples/`` for a more in-depth overview of CyLag's possibilities.
Also see :ref:`Setup` for a complete breakdown of the input parameters (and make sure you read the theoretical aspects to fully understand them).

.. _Python: http://python.org/
.. _Cython: http://cython.org/
.. _openTELEMAC: https://www.opentelemac.org/
