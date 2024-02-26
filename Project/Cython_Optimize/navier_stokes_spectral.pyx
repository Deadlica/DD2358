cimport cython
cimport numpy as cnp
import numpy as np
import pyfftw
# Initialisera pyFFTW för att använda alla tillgängliga trådar
pyfftw.interfaces.cache.enable()
pyfftw.config.NUM_THREADS = pyfftw.detect_number_of_threads()

""""
cnp.import_array()
DTYPE = np.int64
ctypedef cnp.int64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef cnp.ndarray poisson_solve(cnp.ndarray[floatAll, ndim=2] rho, cnp.ndarray[floatAll, ndim=2] kSq_inv):
	cdef cnp.ndarray[floatAll, ndim=2] V_hat = -(np.fft.fftn( rho )) * kSq_inv
	cdef cnp.ndarray[floatAll, ndim=2] V = np.real(np.fft.ifftn(V_hat))
	return V
	"""
@cython.boundscheck(False)
@cython.wraparound(False)
def poisson_solve3(cnp.ndarray[double, ndim=2] rho, cnp.ndarray[double, ndim=2] kSq_inv):
	cdef:
		cnp.ndarray[cnp.complex128_t, ndim=2] V_hat = -np.fft.fftn(rho) * kSq_inv
		cnp.ndarray[cnp.complex128_t, ndim=2] V_org = np.fft.ifftn(V_hat)
		int i, j
		int x = rho.shape[0]
		int y = rho.shape[1]

	cdef cnp.ndarray[cnp.float64_t, ndim=2] V_real= np.empty((x, y), dtype=np.float64)

	for i in range(x):
		for j in range(y):
			V_real[i, j] = V_org[i, j].real
	return V_real

@cython.boundscheck(False)
@cython.wraparound(False)
def diffusion_solve(cnp.ndarray[double, ndim=2] v, float dt, float nu, cnp.ndarray[double, ndim=2] kSq ):
	""" solve the diffusion equation over a timestep dt, given viscosity nu """
	cdef cnp.ndarray[ cnp.complex128_t, ndim=2] v_hat = (np.fft.fftn( v )) / (1.0+dt*nu*kSq)
	cdef cnp.ndarray[ cnp.complex128_t, ndim=2] v_ret = (np.fft.ifftn(v_hat))

	cdef:
		int x = v.shape[0]
		int y = v.shape[1]
	cdef cnp.ndarray[cnp.float64_t, ndim=2] v_return= np.empty((x, y), dtype=np.float64)
	for i in range(x):
		for j in range(y):
			v_return[i, j] = v_ret[i, j].real
			
	return v_return




@cython.boundscheck(False)
@cython.wraparound(False)
def poisson_solve2(cnp.ndarray[cnp.float64_t, ndim=2] rho, cnp.ndarray[cnp.float64_t, ndim=2] kSq_inv):
	cdef int x = rho.shape[0]
	cdef int y = rho.shape[1]

	rho_aligned = pyfftw.empty_aligned((x, y), dtype='complex128')
	V_hat_aligned = pyfftw.empty_aligned((x, y), dtype='complex128')

	rho_aligned[:] = rho.astype('complex128')

	fft_plan = pyfftw.FFTW(rho_aligned, V_hat_aligned, direction='FFTW_FORWARD')
	ifft_plan = pyfftw.FFTW(V_hat_aligned, rho_aligned, direction='FFTW_BACKWARD')

	fft_plan()
	V_hat_aligned *= -kSq_inv
	ifft_plan()

	cdef cnp.ndarray[cnp.float64_t, ndim=2] V_real = np.empty((x, y), dtype=np.float64)

	for i in range(x):
		for j in range(y):
			V_real[i, j] = rho_aligned[i, j].real

	return V_real