import matplotlib.pyplot as plt
import importlib
import sys
import statistics as stat
from timeit import default_timer as timer

param_name = {
    "N" : "Spatial Resolution",
    "dt" : "Timestep",
}

# Default module to run navier stokes with
modules = {
    "numpy" : "navier_stokes_spectral",
    "algorithmic" : "Algorithmic_Optimize.navier_stokes_spectral",
    "cupy" : "Cupy_Optimize.navier_stokes_spectral",
    "pytorch" : "Pytorch_Optimize.navier_stokes_spectral",
}
nss = ""

graphs = {}

# Iterations per variable size
ITERS = 5


def get_measurements(param: str, args: list):
    """
    Computes and measures the execution time of the navier stokes spectral
    while varying `param` with the values in `args`
    :param param: the parameter to be varied in the navier stokes
    spectral function
    :param args: the values that `param` will use for its iterations
    :return: a list of all time measurements
    """
    times = []
    for val in args:
        for _ in range(ITERS):
            start = 0.0
            match param:
                case "N":
                    start = timer()
                    nss.main(N=val)
                case "dt":
                    start = timer()
                    nss.main(dt=val)
                case "nu":
                    start = timer()
                    nss.main(nu=val)
                case _:
                    return
            times.append(timer() - start)
    return times


def get_avg(times: list):
    """
    Computes the average time for each size. The number of
    values per size is the same as `ITERS`
    :param times: list of all time measurements. The size of
    `times` needs to be `len(args) * ITERS`
    :return: list of time averages
    """
    averages = []
    for i in range(int(len(times) / ITERS)):
        averages.append(stat.mean(times[i * ITERS : i * ITERS + ITERS]))
    return averages


def get_std(times: list):
    """
    Computes the standard deviation for each size.
    The number of values per size is the same as `ITERS`
    :param times: list of all time measurements. The size of
    `times` needs to be `len(args) * ITERS`
    :return: list of standard devations
    """
    stds = []
    for i in range(int(len(times) / ITERS)):
        stds.append(stat.stdev(times[i * ITERS : i * ITERS + ITERS]))
    return stds


def get_min(times: list):
    """
    Computes the minimum time for each size. The number of
    values per size is the same as `ITERS`
    :param times: list of all time measurements. The size of
    `times` needs to be `len(args) * ITERS`
    :return: list of the minimum time for each size
    """
    mins = []
    for i in range(int(len(times) / ITERS)):
        mins.append(min(times[i * ITERS : i * ITERS + ITERS]))
    return mins


def get_max(times: list):
    """
    Computes the maximum time for each size. The number of
    values per size is the same as `ITERS`
    :param times: list of all time measurements. The size of
    `times` needs to be `len(args) * ITERS`
    :return: list of the maximum time for each size
    """
    maxs = []
    for i in range(int(len(times) / ITERS)):
        maxs.append(max(times[i * ITERS : i * ITERS + ITERS]))
    return maxs


def plot(x: list, y: list, std: list, param: str, module: str):
    """
    Plots a graph of execution time with varying a variable in
    navier stokes spectral and with standard deviation applied.
    :param x: the size values used for the varying parameter
    :param y: the execution times of navier stokes spectral
    :param std: the standard devations of the execution times
    :param param: the varying parameter used
    :return: None
    """
    if param not in graphs:
        plt.figure()
    plt.title("Varying parameter: " + param_name[param])
    plt.plot(x, y, label=module)
    plt.yscale("log")
    plt.errorbar(x, y, yerr=std, fmt="o", label=module + " std")
    plt.ylabel("Time (log(s))")
    plt.xlabel(param_name[param] + " (" + param + ")")
    plt.legend(loc="best")


if __name__ == "__main__":
    # Initializing variables
    cli_args = []
    arg_vals = {}
    arg_vals["N"]  = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    arg_vals["dt"] = [0.001, 0.002, 0.003, 0.004, 0.005 , 0.006, 0.007, 0.008, 0.009, 0.01]

    args_done = False

    # Default run mode, vary all parameters
    if len(sys.argv) == 1:
        cli_args = ["N", "dt"]
        args_done = True
    # Too many arguments provided
    elif len(sys.argv) > 3:
        print("Excessive number of arguments, please provide at most two.")
        exit(0)

    # parse arguments
    # 1-2 variable parameters are allowed as arguments
    if not args_done:
        for i in range(1, len(sys.argv)):
            match sys.argv[i]:
                case "N":
                    cli_args.append(sys.argv[i])
                case "dt":
                    cli_args.append(sys.argv[i])
                case _:
                    print("Please provide valid argument! [N/dt]")
                    exit(0)
        

    avg, std, minimum, maximum = {}, {}, {}, {}

    # Main loop
    for param in cli_args:
        for module, path in modules.items():
            # Setting module
            nss = importlib.import_module(path)
            times = get_measurements(param, arg_vals[param])
            avg[param] = get_avg(times)
            std[param] = get_std(times)
            minimum[param] = get_min(times)
            maximum[param] = get_max(times)
            plot(arg_vals[param], avg[param], std[param], param, module)
            graphs.update({param : True})
        
    plt.show()
