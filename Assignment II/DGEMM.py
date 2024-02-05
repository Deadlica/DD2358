import numpy as np
from array import array

def DGEMM_np(A: np.array, B: np.array):
    C = np.zeros((len(A), len(B[0])))
    N=len(A)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i][j] = C[i][j] + A[i][k] * B[k][j]
    
    return C

def DGEMM_list(A: list, B: list):
    N=len(A)
    C = [[0.0 for j in range(N)] for i in range(N)]
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i][j] = C[i][j] + A[i][k] * B[k][j]
    
    return C

def DGEMM_array(A: array, B: array, N: int):
    C = array("d", [0.0 for _ in range(N * N)])
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i * N + j] += A[i * N  + k] * B[k * N + j]
    
    return C