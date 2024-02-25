from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize("navier_stokes_spectral.pyx"), compiler_directives={"language_level": "3"})