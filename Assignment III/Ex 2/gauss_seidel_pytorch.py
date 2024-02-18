import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import torch as pt

def gauss_seidel(f, n):
    newf = f.clone()
    
    for _ in range(n):
        newf = 0.25 * (
            pt.roll(newf, 0, +1) +
            pt.roll(newf, 0, -1) +
            pt.roll(newf, +1, 0) +
            pt.roll(newf, -1, 0)
        )
    
    return newf


def plot_perf(data, time):
    fig, ax = plt.subplots()
    
    ax.plot(data, time, label="Computation Time", marker="o") 
    
    ax.set_xlabel("Grid size (N x N)")  
    ax.set_ylabel("Time (s)") 
    ax.set_title("2D Poisson Equation: Performance for Different Grid Sizes") 
    
    ax.legend() 

    plt.show() 
    
if __name__ == "__main__":
    iterations = 1000
    N = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    time_measurements = []
    for matrix_size in N: 
        f = pt.rand(matrix_size, matrix_size)
        f[0, :] = f[-1, :] = f[:, 0] = f[:, -1] = 0.0 

        f.cuda()
        start = timer()
        for i in range(iterations): 
            f = gauss_seidel(f,matrix_size)
        time_measurements.append(timer() - start)        
    
    plot_perf(N,time_measurements)
