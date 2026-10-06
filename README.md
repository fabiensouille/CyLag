# CyLag - Hybrid Eulerian-Lagrangian Particle Tracking Framework

A high-performance Cython-based framework for simulating Lagrangian particle transport and dispersion in fluid flows, with a particular focus on free-surface hydrodynamic applications.

- **Version**: 0.9
- **Author**: Fabien Souillé (EDF R&D, LNHE)
- **First release date**: 2023
- **Language**: Cython
- **License**: GNU GPL v3

---

## Quick installation

### Requirements:

The following packages are required to install and use CyLag: cython >= 0.29, numpy, scipy, mpi4py, h5py, pytest, matplotlib, setuptools, sphinx (for the documentation).
If you already use openTELEMAC, all python requirements should already be installed (except sphinx). 
If not, the recommanded practice is to install a python virtual environment:

```
python3 -m venv pyenv/cylag
```

After creation, activate venv :

```
source pyenv/cylag/bin/activate
```

Install all requirements :

```
pip install -r REQUIREMENTS.txt
```

### Compile from source:

Before using CyLag, the core library must be compiled. 
Compilation is performed by executing the ``setup.py`` script located in the root directory of the repository.
Once compiled, CyLag can be imported and used as a standard Python package.

* Compile CyLag :

```
python3 setup.py build_ext --inplace
```

* Add cylag to python path, copy in ~/.bashrc : 

```
export PYTHONPATH=$PYTHONPATH:</path/to/cylag>
```

* Test importing cylag :

```
python3
import cylag
```

When recompiling the code, you might want to clean the last built extensions, you can do it by using :

```
python3 setup.py clean
```

## Build documentation

The CyLag repository also includes a dedicated Sphinx-based documentation, providing installation instructions, theoretical background, API references, tutorials, and practical examples to facilitate the use and development of the code.


Install Sphinx :

```
pip install sphinx sphinx-gallery sphinx-autodoc-typehints
```

Build documentation : 

```
cd docs
make html
```

The documentation should be available in `docs/build/html/index.html`.

## Useful commands for developpers:

* Run unittests: 

```
python -m pytest tests/*.py
```
