import matplotlib.pyplot as plt
from timeit import default_timer as timer
import cupy as cp

def jacobi(f):
    newf = f.copy()
    sr = cp.roll(newf, 1, 1)[1:-1, 1:-1]
    sl = cp.roll(newf, -1, 1)[1:-1, 1:-1]
    sd = cp.roll(newf, 1, 0)[1:-1, 1:-1]
    su = cp.roll(newf, -1, 0)[1:-1, 1:-1]
    newf[1:-1, 1:-1] = 0.25 * (sr + sl + sd + su)
    return newf


def plot_perf(data, time):
    fig, ax = plt.subplots()
    ax.plot(data, time, label="Computation Time", marker="o")
    ax.set_xlabel("Grid size (N x N)")
    ax.set_ylabel("Time (s)")
    ax.set_title("2D Poisson Equation: Performance for Different Grid Sizes")

    ax.legend()
    plt.show()

def get_measurements(N, iterations):
    time_measurements = []
    for matrix_size in N:
        f = cp.random.rand(matrix_size, matrix_size)
        f[0, :] = f[-1, :] = f[:, 0] = f[:, -1] = 0.0

        start = timer()
        for i in range(iterations):
            f = jacobi(f)
        cp.cuda.Stream.null.synchronize()
        time_measurements.append(timer() - start)

    return time_measurements


if __name__ == "__main__":
    iterations = 1000
    N = [1000, 2000, 3000, 4000, 5000]
    time_measurements = get_measurements(N, iterations)
    plot_perf(N, time_measurements)
