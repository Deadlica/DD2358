import sys
sys.path.append("./") 
sys.path.append("../") 
import test_numpy 
import unittest
import cupy as cp
import Cupy_Optimize.navier_stokes_spectral as nss

class TestFluidDynamicsSolvers(unittest.TestCase):
    """
    This class tests the functionality and accuracy of fluid dynamics solvers,
    specifically focusing on spectral methods for solving Navier-Stokes equations.
    It compares the output of optimized CuPy implementations against original NumPy implementations.
    """
    def setUp(self):
        """
        Set up the test environment before each test method is executed.
        Initializes spatial domain parameters, wave numbers, viscosity, time step, and initial conditions.
        """
        self.N = 64
        self.dx = 2.0 * cp.pi / self.N
        k = cp.fft.fftfreq(self.N, d=self.dx)
        self.kx, self.ky = cp.meshgrid(k, k, indexing='ij')
        self.kSq = self.kx**2 + self.ky**2
        self.kSq_inv = cp.zeros_like(self.kSq)
        self.kSq_inv[self.kSq > 0] = 1.0 / self.kSq[self.kSq > 0]
        self.klin = 2.0 * cp.pi / 1 * cp.arange(-400/2, 400/2)
        self.nu = 0.1
        self.dt = 0.01
        self.xlin = cp.linspace(0,1, num=400+1)
        self.xlin = self.xlin[0:self.N] 
        self.kmax = cp.max(self.klin)
        self.dealias = (cp.abs(self.kx) < (2./3.)*self.kmax) & (cp.abs(self.ky) < (2./3.)*self.kmax)
        self.rho = cp.random.rand(self.N, self.N)
        self.v = cp.random.rand(self.N, self.N)
        self.xx, self.yy = cp.meshgrid(self.xlin, self.xlin)
        self.vx = -cp.sin(2*cp.pi*self.yy)
        self.vy =  cp.sin(2*cp.pi*self.xx*2) 
        self.dvx_x, self.dvx_y = nss.grad(self.vx, self.kx, self.ky)
        self.dvy_x, self.dvy_y = nss.grad(self.vy, self.kx, self.ky)
        self.rhs_x = -(self.vx * self.dvx_x + self.vy * self.dvx_y)
        self.rhs_y = -(self.vx * self.dvy_x + self.vy * self.dvy_y)
        self.div_rhs = nss.div(self.rhs_x, self.rhs_y, self.kx, self.ky)
        self.P=nss.poisson_solve( self.div_rhs, self.kSq_inv )
        #create numpy class
        self.numpy_data = test_numpy.FluidDynamicsSolversNumpy()

        # Both rho's are random, we make them the same to ensure
        # test_poisson_solve works
        for i in range(len(self.rho)):
            for j in range(len(self.rho[i])):
                self.numpy_data.rho[i][j] = self.rho[i][j]
    

    def test_poisson_solve(self):
        """
        Test the Poisson solver for accuracy by comparing the optimized solver's output against the original solver's output.
        """
        expected_ans = test_numpy.original_nss.poisson_solve(self.numpy_data.rho, self.numpy_data.kSq_inv)
        optimized_ans = nss.poisson_solve(self.rho, self.kSq_inv)
        self.assertTrue(cp.allclose(expected_ans, optimized_ans))


    def test_diffusion_solve(self):
        """
        Test the diffusion solver for accuracy by comparing the optimized solver's output against the original solver's output.
        """
        expected_ans = test_numpy.original_nss.diffusion_solve(self.numpy_data.vx, self.numpy_data.dt, self.numpy_data.nu, self.numpy_data.kSq)
        optimized_ans = nss.diffusion_solve(self.vx, self.dt, self.nu, self.kSq)
        self.assertTrue(cp.allclose(expected_ans, optimized_ans))


    def test_curl_solve(self):
        """
        Test the curl calculation for accuracy by comparing the optimized solver's output against the original solver's output.
        """
        expected_ans = test_numpy.original_nss.curl(self.numpy_data.vx, self.numpy_data.vy, self.numpy_data.kx, self.numpy_data.ky)
        optimized_ans = nss.curl(self.vx, self.vy, self.kx, self.ky)
        self.assertTrue(cp.allclose(expected_ans, optimized_ans))


    def test_div_solve(self):
        """
        Test the div calculation for accuracy by comparing the optimized solver's output against the original solver's output.
        """
        expected_ans = test_numpy.original_nss.div(self.numpy_data.rhs_x, self.numpy_data.rhs_y, self.numpy_data.kx, self.numpy_data.ky)
        optimized_ans = nss.div(self.rhs_x, self.rhs_y, self.kx, self.ky)
        self.assertTrue(cp.allclose(expected_ans, optimized_ans))


    def test_grad_solve(self):
        """
        Test the grad calculation for accuracy by comparing the optimized solver's output against the original solver's output.
        """
        expected_ans = test_numpy.original_nss.grad(self.numpy_data.P, self.numpy_data.kx, self.numpy_data.ky)
        optimized_ans = nss.grad(self.P, self.kx, self.ky)
        self.assertTrue(cp.allclose(expected_ans, optimized_ans))


if __name__ == '__main__':
    unittest.main()
