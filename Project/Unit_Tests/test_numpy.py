#import sys
import numpy as np
#sys.path.append('../') 
#import Orgin_navier_stokes_spectral as original_nss
class TestFluidDynamicsSolversNumpy():

    def setUp(self):
        self.N = 64
        self.dx = 2.0 * np.pi / self.N
        k = np.fft.fftfreq(self.N, d=self.dx)
        self.kx, self.ky = np.meshgrid(k, k, indexing='ij')
        self.kSq = self.kx**2 + self.ky**2
        self.kSq_inv = np.zeros_like(self.kSq)
        self.kSq_inv[self.kSq > 0] = 1.0 / self.kSq[self.kSq > 0]
        self.klin = 2.0 * np.pi / 1 * np.arange(-400/2, 400/2)
        self.nu = 0.1
        self.dt = 0.01
        self.xlin = np.linspace(0,1, num=400+1)
        self.xlin = self.xlin[0:self.N] 
        self.kmax = np.max(self.klin)
        self.dealias = (np.abs(self.kx) < (2./3.)*self.kmax) & (np.abs(self.ky) < (2./3.)*self.kmax)
        self.rho = np.random.rand(self.N, self.N)
        self.v = np.random.rand(self.N, self.N)
        self.xx, self.yy = np.meshgrid(self.xlin, self.xlin)
        self.vx = -np.sin(2*np.pi*self.yy)
        self.vy =  np.sin(2*np.pi*self.xx*2) 
        self.dvx_x, self.dvx_y = original_nss.grad(self.vx, self.kx, self.ky)
        self.dvy_x, self.dvy_y = original_nss.grad(self.vy, self.kx, self.ky)
        self.rhs_x = -(self.vx * self.dvx_x + self.vy * self.dvx_y)
        self.rhs_y = -(self.vx * self.dvy_x + self.vy * self.dvy_y)
        self.rhs_x = original_nss.apply_dealias(self.rhs_x, self.dealias)
        self.rhs_y = original_nss.apply_dealias(self.rhs_y, self.dealias)
        self.div_rhs = original_nss.div(self.rhs_x, self.rhs_y, self.kx, self.ky)
        self.P=original_nss.poisson_solve( self.div_rhs, self.kSq_inv )
        self.vx += self.dt * self.rhs_x
        self.vy += self.dt * self.rhs_y


