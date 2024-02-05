from timeit import default_timer as timer
from array import array
import matplotlib.pyplot as plt
import numpy as np
import sys

STREAM_ARRAY_SIZE = 1000
STREAM_ARRAY_TYPE = "f"
TESTS = 4

def init_lists():
    a = [1.0 for _ in range(STREAM_ARRAY_SIZE)]
    b = [2.0 for _ in range(STREAM_ARRAY_SIZE)]
    c = [0.0 for _ in range(STREAM_ARRAY_SIZE)]
    return [a, b, c]

def init_arrays():
    lists = init_lists()
    a = array(STREAM_ARRAY_TYPE, lists[0])
    b = array(STREAM_ARRAY_TYPE, lists[1])
    c = array(STREAM_ARRAY_TYPE, lists[2])
    return [a, b, c]

def stream_benchmark(a, b, c, scalar: float):
    times = [0.0 for _ in range(4)]
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

def compute_moved_data():
    size = 0
    if STREAM_ARRAY_TYPE == "f":
        size = 4
    elif STREAM_ARRAY_TYPE == "d":
        size = 8
    copy = 2 * size  * STREAM_ARRAY_SIZE
    add = 2 * size  * STREAM_ARRAY_SIZE
    scale = 3 * size  * STREAM_ARRAY_SIZE
    triad = 3 * size  * STREAM_ARRAY_SIZE
    return copy, add, scale, triad

if __name__ == "__main__":
    vals = []
    if len(sys.argv) != 2:
        print("Please provide 1 argument. [list/array]")
        exit(0)
    elif sys.argv[1] != "list" and sys.argv[1] != "array":
        print("Please provide a valid argument! [list/array]")
        exit(0)

    copy_bw = [0.0 for _ in range(TESTS)]
    add_bw = [0.0 for _ in range(TESTS)]
    scale_bw = [0.0 for _ in range(TESTS)]
    triad_bw = [0.0 for _ in range(TESTS)]

    x_values = []
    for i in range(TESTS):
        if sys.argv[1] == "list":
            vals = init_lists()
        elif sys.argv[1] == "array":
            vals = init_arrays()
        a = vals[0]
        b = vals[1]
        c = vals[2]
        scalar = 2.0
        t_copy, t_add, t_scale, t_triad = stream_benchmark(a, b, c, scalar)
        m_copy, m_add, m_scale, m_triad = compute_moved_data()
        copy_bw[i] = m_copy / t_copy
        add_bw[i] = m_add / t_add
        scale_bw[i] = m_scale / t_scale
        triad_bw[i] = m_triad / t_triad
        x_values.append(STREAM_ARRAY_SIZE)
        STREAM_ARRAY_SIZE *= 10

x = np.arange(len(x_values))
width = 0.2
fig, ax = plt.subplots()

bars1 = ax.bar(x - 3*width/2, copy_bw, width, label="Copy")
bars2 = ax.bar(x - width/2, add_bw, width, label="Add")
bars3 = ax.bar(x + width/2, scale_bw, width, label="Scale")
bars4 = ax.bar(x + 3*width/2, triad_bw, width, label="Triad")

ax.set_xlabel("STREAM Array Size")
ax.set_ylabel("Memory Bandwidth (B/s)")
ax.set_title("STREAM Benchmark for " + sys.argv[1] + "s")
ax.set_xticks(x)
ax.set_xticklabels(x_values)
ax.legend()

plt.show()