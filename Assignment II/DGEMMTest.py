from timeit import default_timer as timer
import pytest
from DGEMM import *
import numpy as np
from array import array
import sys

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
        return (A, B, C, N)

def get(N: int, tests: int, type: str):
        data = list()
        if type == "np":
            for _ in range(tests):
                data.append(get_np(N))
                N *= 2
        if type == "list":
            for _ in range(tests):
                data.append(get_list(N))
                N *= 2
        if type == "array":
            for _ in range(tests):
                data.append(get_array(N))
                N *= 2
        return data

@pytest.mark.parametrize('A, B, expected', get(3, 6, "np"))
def test_np(A, B, expected):
        assert np.array_equal(DGEMM_np(A, B), expected)

@pytest.mark.parametrize('A, B, expected', get(3, 6, "list"))
def test_list(A, B, expected):
        assert np.array_equal(DGEMM_list(A, B), expected)

@pytest.mark.parametrize('A, B, expected, N', get(3, 6, "array"))
def test_array(A, B, expected, N):
        assert np.array_equal(DGEMM_array(A, B, N), expected)


# perform 5 tests of size 3 with lists, get average, deviation etc
# perform 5 tests of size 6 with lists, get average, deviation etc
# perform 5 tests of size 3 with array, get average, deviation etc
# perform 5 tests of size 6 with array, get average, deviation etc
def get_data(N: int, tests: int, type: str):
        data = list()
        if type == "np" or type == "matmul":
            for _ in range(tests):
                data.append(get_np(N))
        if type == "list":
            for _ in range(tests):
                data.append(get_list(N))
        if type == "array":
            for _ in range(tests):
                data.append(get_array(N))
        return data


def execution (N: int, tests: int, type: str):
        data = get_data(N, tests, type)
        times = []
        for i in data:
                if type == "array":
                        start = timer()
                        DGEMM_array(i[0], i[1], i[3])
                        duration = timer() - start
                elif type == "list":
                        start = timer()
                        DGEMM_list(i[0], i[1])
                        duration = timer() - start
                elif type == "np":
                        start = timer()
                        DGEMM_np(i[0], i[1])
                        duration = timer() - start
                elif type == "matmul":
                        start = timer()
                        np.matmul(i[0], i[1])
                        duration = timer() - start
                times.append(duration)
        print_execution(times, tests, N, type)
        

       
def print_execution(times: list, tests: int, N: int, type: str):
        avg = np.mean(times)
        std = np.std(times)
        if type == "matmul":
                print("Tests: " + str(tests) + "\tSize: " + str(N) + "\t\tDatatype: " + type + "\tMean: " + str(avg) + "\t\tStd: " + str(std))
        else:
                print("Tests: " + str(tests) + "\tSize: " + str(N) + "\t\tDatatype: " + type + "\t\tMean: " + str(avg) + "\t\tStd: " + str(std))


if __name__ == "__main__":
       #Write code for measuring execution time here
        for i in range(3, 12, 3):
                execution(i, 5, "array")
                execution(i, 5, "list")
                execution(i, 5, "np")
                execution(i, 5, "matmul")
                print("="*120)