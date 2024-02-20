import matplotlib.pyplot as plt
import jacobi_pytorch as torch
import jacobi_cupy as cupy
import gauss_seidel as cpu

def plot_perf(data, cupy_times, torch_times, cpu_times):
    fig, ax = plt.subplots()
    ax.plot(data, cupy_times, label="CuPy Computation Time", marker="o")
    ax.plot(data, torch_times, label="Pytorch Computation Time", marker="o")
    ax.plot(data, cpu_times, label="CPU Computation Time", marker="o")
    ax.set_xlabel("Grid size (N x N)")
    ax.set_ylabel("Time (s)")
    ax.set_title("2D Poisson Equation: Performance for Different Grid Sizes")

    ax.legend()
    plt.show()

if __name__ == "__main__":
    iterations = 1000
    data = [1000, 1200, 1400, 1600, 1800, 2000]
    cupy_times = cupy.get_measurements(data, iterations)
    torch_times = torch.get_measurements(data, iterations)
    cpu_times = cpu.get_measurements(data, 10)
    plot_perf(data, cupy_times, torch_times, cpu_times)