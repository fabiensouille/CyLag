.. _dev:

******************************************************************************
Developer guide
******************************************************************************

Get access
==============================================================================

To get developer access to the repository, send an email to: `fabien.souille@edf.fr <fabien.souille@edf.fr>`_

Development checklist
==============================================================================

Before submitting any changes to CyLag, follow these simple steps:

* Create development branch and make your modifications;
* Compile the code in debug mode and run unit-tests to check non-regression:

.. code-block:: bash

  python3 setup.py clean 
  python3 setup.py build_ext --inplace --debug
  python3 -m pytest tests/*.py

* Compile the code in standard mode and run unit-tests to check non-regression:

.. code-block:: bash

  python3 setup.py clean 
  python3 setup.py build_ext --inplace
  python3 -m pytest tests/*.py

* Compile the Sphinx documentation and check the gallery of examples: 

.. code-block:: bash

  cd docs && make html

* Create a new example and documentation to validate your development;
* Create a merge request.

.. note::
  Verify functionality, code quality, test coverage, documentation updates, dependency management, performance impact, and deployment readiness before submitting any changes.


Unit tests
==============================================================================

After any modification of the source code and after recompiling, you should run unit tests to check non-regression.
To run all unit-tests at once, type:

.. code-block:: bash

  python3 -m pytest tests/*.py


You can also run unit tests of specific modules:

.. code-block:: bash

  python3 -m pytest tests/test_00_geom_mesh.py
  python3 -m pytest tests/test_14_core_solver.py
  python3 -m pytest tests/test_20_lsm1_analytical_advection_2d.py

The test categories are organized as follows:

- ``test_0X``: Geometry and mesh operations
- ``test_1X``: Core classes
- ``test_2X``: LSM-1 verifications
- ``test_3X``: LSM-2 verifications
- ``test_4X``: LSM-3 / RNG verifications
- ``test_5X``: Extra modules
- ``test_6X``: Parallel functionality
- ``test_7X``: Performance benchmarks


Performance considerations and profiling
==============================================================================

Memory layout:

- **Contiguous arrays**: All particle data stored in C-contiguous NumPy arrays,
- **Pool allocation**: Reduces dynamic memory allocation overhead,
- **Temporary arrays**: Field values cached at current time step.

Computational hotspots:

1. **Point Localization** (~30-40% of runtime)
2. **Field Interpolation** (~20-30% of runtime)
3. **Time Integration** (~15-25% of runtime)
4. **Random Number Generation** (~5-10% for diffusive cases)

Parallelization strategy:

- **Domain decomposition**: Not used (mesh not partitioned),
- **Particle decomposition**: Particles distributed among processes,
- **Communication**: Only for result merging at end.


To check the performance of the code after a modification, you can use the profiling options of Cython.
To do so, you first need to activate profiling (and recompile):

.. code-block:: bash

  ./profile.sh 1


To deactivate profiling:

.. code-block:: bash

  ./profile.sh 0
