import numpy as np
import matplotlib.pyplot as plt
import cythonfn
from timeit import default_timer as timer

def gauss_seidel_func(f, n):
    newf = f.copy()

    for i in range(1,newf.shape[0]-1):
        for j in range(1,newf.shape[1]-1):
            newf[i,j] = 0.25 * (newf[i,j+1] + newf[i,j-1] + newf[i+1,j] + newf[i-1,j])
    return newf


def plot_perf(data, time):
    fig, ax = plt.subplots()

    ax.plot(data, time, label='Computation Time',marker='o' )
    ax.set_xlabel("Grid size (N by N)")
    ax.set_ylabel("Time (s)")
    ax.set_title("2D Poisson Equation: Performance for Different Grid Sizes")
    
    ax.legend()
    plt.show()


if __name__ == "__main__":
    iterations = 1000
    N = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    time_measurements = []
    for matrix_size in N:
        f = np.random.rand(matrix_size,matrix_size)
        f[0, :] = f[-1, :] = f[:, 0] = f[:, -1] = 0.0 

        start = timer()
        for i in range(iterations):
            f = cythonfn.gauss_seidel_func(f, matrix_size)
        time_measurements.append(timer() - start)
    
    plot_perf(N, time_measurements)