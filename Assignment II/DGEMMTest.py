import pytest
from DGEMM import *
import numpy as np
from array import array

def matrix(N, type="list"):
        if type != "np" and type != "list" and type != "array":
                print("Bad input")
                exit(0)
        low, high = 0, 100
        if type == "np":
                return np.array([[np.random.randint(low, high) for j in range(N)] for i in range(N)])
        elif type == "list":
                return [[np.random.randint(low, high) for j in range(N)] for i in range(N)]
        return array("d", [np.random.randint(low, high) for _ in range(N * N)])

def get_np(N: int):
        A = matrix(N, "np")
        B = matrix(N, "np")
        C = np.matmul(A, B)
        return (A, B, C)
        
def get_list(N: int):
        A = matrix(N, "list")
        B = matrix(N, "list")
        C = np.matmul(A, B)
        return (A, B, C)
        
def get_array(N: int):
        A = matrix(N, "array")
        B = matrix(N, "array")
        A_list = [[A[i * N + j] for j in range(N)] for i in range(N)]
        B_list = [[B[i * N + j] for j in range(N)] for i in range(N)]
        C_list = np.matmul(A_list, B_list)
        C = array("d", [C_list[j][i] for j in range(N) for i in range(N)])
        return (A, B, C)

def get(N: int, tests: int, type: str):
        data = list()
        if type == "np":
            for _ in range(tests):
                data.append(get_np(N))
                N *= 10
        if type == "list":
            for _ in range(tests):
                data.append(get_list(N))
                N *= 10
        if type == "array":
            for _ in range(tests):
                data.append(get_array(N))
                N *= 10
        return data

@pytest.mark.parametrize('A, B, expected', get(5, 1, "np"))
def test_np(A, B, expected):
        assert np.array_equal(DGEMM_np(A, B), expected)

@pytest.mark.parametrize('A, B, expected', get(5, 1, "list"))
def test_list(A, B, expected):
        assert np.array_equal(DGEMM_list(A, B), expected)

@pytest.mark.parametrize('A, B, expected', get(5, 1, "array"))
def test_array(A, B, expected):
        assert np.array_equal(DGEMM_array(A, B, 5), expected)