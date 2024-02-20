from timeit import default_timer as timer
cimport cython

cdef unsigned int STREAM_ARRAY_SIZE = 1000

@cython.boundscheck(False)
def stream_benchmark(float[:] a, float[:] b, float[:] c, float scalar):
    cdef :
        unsigned int i
        double times[3]
    # copy
    times[0] = timer()
    for i in range(STREAM_ARRAY_SIZE):
        c[i] = a [i]
    times[0] = timer() - times[0]

    # scale
    times[1] = timer()
    for i in range(STREAM_ARRAY_SIZE):
        b[i] = scalar * c[i]
    times[1] = timer() - times[1]

    # add
    times[2] = timer()
    for i in range(STREAM_ARRAY_SIZE):
        c[i] = a[i] + b[i]
    times[2] = timer() - times[2]

    # triad
    times[3] = timer()
    for i in range(STREAM_ARRAY_SIZE):
        a[i] = b[i] + scalar * c[i]
    times[3] = timer() - times[3]

    return times[0], times[2], times[1], times[3]
