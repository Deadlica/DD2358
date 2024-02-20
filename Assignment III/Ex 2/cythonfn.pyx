cimport cython

@cython.boundscheck(False)
def gauss_seidel_func(double [:,:] f, int n):
    cdef double [:,:] newf = f.copy()
    cdef unsigned int i, j
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            newf[i][j] = 0.25 *  newf[i][j + 1] + newf[i][j - 1] + newf[i + 1][j] + newf[i - 1][j]
    return newf