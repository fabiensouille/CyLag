# coding: utf-8
"""
Project name: CyLag
Copyright (C) 2023 Fabien Souille

This program is free: you can redistribute it and/or modify it
under the terms of the GNU General Public License published by the Free Software Foundation,
either version 3 of the license, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

See the GNU General Public License for details.

You should have received a copy of the GNU General Public License
with this program. If not, see <https://www.gnu.org/licenses/>.
"""
import sys
import os
import shutil
import glob
from pathlib import Path
from setuptools import find_packages, Command
from distutils.core import setup
from distutils.extension import Extension

# ~~~~~~~~~~~~~~~~~~~~
# Cython build options
# ~~~~~~~~~~~~~~~~~~~~
# check if the command is "clean" to avoid importing Cython when cleaning
is_clean_command = "clean" in sys.argv[1:]

# check if the command is "debug" to set compiler directives accordingly
is_debug_build = "--debug" in sys.argv[1:]

# Debug compilation mode
# ----------------------
if is_debug_build:
    compiler_directives = {
        'boundscheck': True,
        'wraparound': True,
        'initializedcheck': True,
        "nonecheck": True,
        'cdivision': True,
        'language_level': 3,
    }
    extra_compile_args = [
      "-O1", 
      "-g", 
      "-Wall",
      "-Wextra",
      "-Wuninitialized",
      ]
    extra_link_args = []

# Standard compilation mode
# -------------------------
else:
    compiler_directives = {
        'boundscheck': True,
        'wraparound': True,
        'initializedcheck': False,
        "nonecheck": False,
        'cdivision': True,
        'language_level': 3,
    }
    extra_compile_args = [
        "-O2",
        ]
    extra_link_args = []

if not is_clean_command:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext

# ~~~~~~~~~~~~~~~~~~~~~
# List all dependencies
# ~~~~~~~~~~~~~~~~~~~~~
install_requires = ["cython>=0.29", "numpy", "scipy", "mpi4py", "h5py", "pytest", "matplotlib"]

# ~~~~~~~~~~~
# Description
# ~~~~~~~~~~~
with open("README.md", "r") as fh:
    long_description = fh.read()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# List all extensions of all packages to build
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def list_package_extensions(package_paths, filenames):
    """Create a Extension object of all .pyx file in a package"""
    extensions = []
    for filename in filenames:
        filepath = os.path.join(*(package_paths + [filename]))
        module = os.path.splitext(filename)[0]
        module_path = package_paths + [module,]
        ext = Extension(
            '.'.join(module_path),
            sources=[filepath],
            include_dirs=['.'],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
        extensions.append(ext)
    return extensions

# Extensions
ext_core = list_package_extensions(
    package_paths = ['cylag', 'core'],
    filenames = [
        'constants.pyx',
        'parameters.pyx',
        'eulerian_field_set.pyx',
        'particle_distribution.pyx',
        'lagrangian_particle_set.pyx',
        'simutime.pyx',
        'solver.pyx',
        'solver_parallel.pyx',
    ])
ext_geom = list_package_extensions(
    package_paths = ['cylag', 'geom'],
    filenames = [
        'utils.pyx',
        'intersections.pyx',
        'read_cli.pyx',
        'triangular_mesh.pyx',
        'xylocalizer.pyx',
        'zlocalizer.pyx',
    ])
ext_lsm = list_package_extensions(
    package_paths = ['cylag', 'lsm'],
    filenames = [
        'random_utils.pyx',
        'init_lsm2_fluctuations.pyx',
        'init_lsm3_fluctuations.pyx',
        'update_state_test.pyx',
        'update_state_lsm1.pyx',
        'update_state_lsm2.pyx',
        'update_state_lsm3.pyx',
    ])
ext_io = list_package_extensions(
    package_paths = ['cylag', 'io'],
    filenames = [
        'particles_io.pyx',
        'writer_txt.pyx',
        'writer_vtk.pyx',
        'write_particles.pyx',
        'write_statistics.pyx',
        'parallel_tags.pyx',
        'converter.pyx',
    ])
ext_bnd = list_package_extensions(
    package_paths = ['cylag', 'boundary_conditions'],
    filenames = [
        'compute_bc.pyx',
    ])
ext_extra = list_package_extensions(
    package_paths = ['cylag', 'extra'],
    filenames = [
        'initial_conditions.pyx',
        'scheduler.pyx',
        'particle_source.pyx',
        'plot_utils.pyx',
        'ghost_meshgrid.pyx',
        'statistics_from_particles.pyx',
        'control_shapes.pyx',
    ])
ext_mod = list_package_extensions(
    package_paths = ['cylag', 'modules'],
    filenames = [
        'custom_particle_set.pyx',
        'oil_slick.pyx',
        'jellyfish.pyx',
        'fish.pyx',
        'algae.pyx',
        'macrophyte.pyx',
    ])
extensions = ext_core + ext_geom + ext_lsm\
    + ext_io + ext_bnd + ext_extra + ext_mod

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# custom clean function of built extensions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class CleanCommand(Command):
    description = "Remove Cython build artifacts"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        # Remove build directories
        for dirname in ("build", "dist"):
            shutil.rmtree(dirname, ignore_errors=True)

        # Remove generated files for each extension
        for ext in extensions:
            for source in ext.sources:
                source = Path(source)

                # Only process pyx sources
                if source.suffix != ".pyx":
                    continue

                # Generated C/C++ files
                for generated in (
                    source.with_suffix(".c"),
                    source.with_suffix(".cpp"),):
                    if generated.exists():
                        print(f"Removing {generated}")
                        generated.unlink()

                # Generated shared library (*.so, *.pyd)
                module_name = source.stem

                for pattern in (
                    f"{module_name}*.so",
                    f"{module_name}*.pyd",):
                    for artifact in source.parent.glob(pattern):
                        print(f"Removing {artifact}")
                        artifact.unlink()

# ~~~~~
# Setup
# ~~~~~
setup(
    name="cylag",
    version='1.0',
    packages=find_packages(include=[\
        "cylag",
        "cylag.core",
        "cylag.lsm",
        "cylag.geom",
        "cylag.io",
        "cylag.boundary_conditions",
        "cylag.extra",
        "cylag.modules",
        ]),
    python_requires=">=3.7",
    install_requires=install_requires,
    description="CyLag : a Cython code for hybrid Eulerian-Lagrangian particle dynamics modeling in free surface flows",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.pleiade.edf.fr/codes-iguasou/cylag.git",
    cmdclass=({"clean": CleanCommand} if is_clean_command else
              {'build_ext': build_ext}),
    ext_modules=(extensions if is_clean_command else cythonize(extensions,
        quiet=False,
        force=False,
        compiler_directives=compiler_directives,
        )),
    )
