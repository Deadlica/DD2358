# test_calculator.py

import unittest
import sys
sys.path.append('../')
import Cupy_Optimize.navier_stokes_spectral as nss
import numpy as np

class TestFluidDynamicsSolvers(unittest.TestCase):

    def setUp(self):
        self.N = 64
        self.dx = 2.0 * np.pi / self.N
        k = np.fft.fftfreq(self.N, d=self.dx)
        self.kx, self.ky = np.meshgrid(k, k, indexing='ij')
        self.kSq = self.kx**2 + self.ky**2
        self.kSq_inv = np.zeros_like(self.kSq)
        self.kSq_inv[self.kSq > 0] = 1.0 / self.kSq[self.kSq > 0]
        self.nu = 0.1
        self.dt = 0.01
        self.rho = np.random.rand(self.N, self.N)
        self.v = np.random.rand(self.N, self.N)


if __name__ == '__main__':
    unittest.main()
