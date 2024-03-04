import sys
sys.path.append("./") 
sys.path.append("../") 
import numpy as np
import navier_stokes_spectral as original_nss

class FluidDynamicsSolversNumpy():
    """A class for the init state of fluid dynamics solvers focusing on numpy-based implementations.
    """
    N = 64
    dx = 2.0 * np.pi / N
    k = np.fft.fftfreq(N, d=dx)
    kx, ky = np.meshgrid(k, k, indexing='ij')
    kSq = kx**2 + ky**2
    kSq_inv = np.zeros_like(kSq)
    kSq_inv[kSq > 0] = 1.0 / kSq[kSq > 0]
    klin = 2.0 * np.pi / 1 * np.arange(-400/2, 400/2)
    nu = 0.1
    dt = 0.01
    xlin = np.linspace(0,1, num=400+1)
    xlin = xlin[0:N] 
    kmax = np.max(klin)
    dealias = (np.abs(kx) < (2./3.)*kmax) & (np.abs(ky) < (2./3.)*kmax)
    rho = np.random.rand(N, N)
    v = np.random.rand(N, N)
    xx, yy = np.meshgrid(xlin, xlin)
    vx = -np.sin(2*np.pi*yy)
    vy =  np.sin(2*np.pi*xx*2) 
    dvx_x, dvx_y = original_nss.grad(vx, kx, ky)
    dvy_x, dvy_y = original_nss.grad(vy, kx, ky)
    rhs_x = -(vx * dvx_x + vy * dvx_y)
    rhs_y = -(vx * dvy_x + vy * dvy_y)
    div_rhs = original_nss.div(rhs_x, rhs_y, kx, ky)
    P=original_nss.poisson_solve( div_rhs, kSq_inv )

