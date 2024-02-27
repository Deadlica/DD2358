import unittest
import sys
import test_numpy 

import torch as tp
import numpy as np
sys.path.append('../') 
import Pytorch_Optimize.navier_stokes_spectral as nss
import Orgin_navier_stokes_spectral as original_nss
class TestFluidDynamicsSolvers(unittest.TestCase):
    """
    This class tests the functionality and accuracy of fluid dynamics solvers,
    specifically focusing on spectral methods for solving Navier-Stokes equations.
    It compares the output of optimized PyTorch implementations against original NumPy implementations.
    """
    def setUp(self):
        """
        Set up the test environment before each test method is executed.
        Initializes spatial domain parameters, wave numbers, viscosity, time step, and initial conditions.
        """
        self.numpy_data = test_numpy.TestFluidDynamicsSolversNumpy()

        

    def test_poisson_solve(self):
        """
        Test the Poisson solver for accuracy by comparing the optimized solver's output against the original solver's output.
        """
        expected_ans = original_nss.poisson_solve(self.numpy_data.rho, self.numpy_data.kSq_inv)
        optimized_ans = nss.poisson_solve(tp.from_numpy(self.numpy_data.rho), tp.from_numpy(self.numpy_data.kSq_inv))
        self.assertTrue(np.allclose(expected_ans, optimized_ans))


    def test_diffusion_solve(self):
        """
        Test the diffusion solver for accuracy by comparing the optimized solver's output against the original solver's output.
        """
        expected_ans = original_nss.diffusion_solve(self.numpy_data.vx, self.numpy_data.dt, self.numpy_data.nu, self.numpy_data.kSq)
        optimized_ans = nss.diffusion_solve(tp.from_numpy(self.numpy_data.vx), self.numpy_data.dt, self.numpy_data.nu, self.numpy_data.kSq)
        self.assertTrue(np.allclose(expected_ans, optimized_ans))


    def test_curl_solve(self):
        """
        Test the curl calculation for accuracy by comparing the optimized solver's output against the original solver's output.
        """
        expected_ans = original_nss.curl(self.numpy_data.vx, self.numpy_data.vy, self.numpy_data.kx, self.numpy_data.ky)
        optimized_ans = nss.curl(tp.from_numpy(self.numpy_data.vx), tp.from_numpy(self.numpy_data.vy), tp.from_numpy(self.numpy_data.kx), tp.from_numpy(self.numpy_data.ky))
        self.assertTrue(np.allclose(expected_ans, optimized_ans))


    def test_div_solve(self):
        """
        Test the div calculation for accuracy by comparing the optimized solver's output against the original solver's output.
        """
        expected_ans = original_nss.div(self.numpy_data.rhs_x, self.numpy_data.rhs_y, self.numpy_data.kx, self.numpy_data.ky)
        optimized_ans = nss.div(tp.from_numpy(self.numpy_data.rhs_x), tp.from_numpy(self.numpy_data.rhs_y), tp.from_numpy(self.numpy_data.kx), tp.from_numpy(self.numpy_data.ky))
        self.assertTrue(np.allclose(expected_ans, optimized_ans))


    def test_grad_solve(self):
        """
        Test the grad calculation for accuracy by comparing the optimized solver's output against the original solver's output.
        """
        expected_ans = original_nss.grad(self.numpy_data.P, self.numpy_data.kx, self.numpy_data.ky)
        optimized_ans = nss.grad(tp.from_numpy(self.numpy_data.P), tp.from_numpy(self.numpy_data.kx), tp.from_numpy(self.numpy_data.ky))
        self.assertTrue(np.allclose(expected_ans, optimized_ans))


if __name__ == '__main__':
    unittest.main()
