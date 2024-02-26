remove:
	rm -rf build

build: remove
	python3 setup.py build_ext --inplace

run:
	python3 navier_stokes_spectral.py

html:
	cython -a navier_stokes_spectral.pyx
