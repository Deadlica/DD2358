import navier_stokes_spectral as nss
import matplotlib.pyplot as plt
import sys
import statistics as stat
from timeit import default_timer as timer

ITERS = 5
param_name = {
    "N" : "Spatial Resolution",
    "dt" : "Timestep",
    "nu" : "Viscosity",
}

#figure, axis = plt.subplots(3, 1)

def get_measurements(param: str, args: list):
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
    averages = []
    for i in range(int(len(times) / ITERS)):
        averages.append(stat.mean(times[i * ITERS : i * ITERS + ITERS]))
    return averages


def get_std(times: list):
    stds = []
    for i in range(int(len(times) / ITERS)):
        stds.append(stat.stdev(times[i * ITERS : i * ITERS + ITERS]))
    return stds


def get_min(times: list):
    mins = []
    for i in range(int(len(times) / ITERS)):
        mins.append(min(times[i * ITERS : i * ITERS + ITERS]))
    return mins

def get_max(times: list):
    maxs = []
    for i in range(int(len(times) / ITERS)):
        maxs.append(max(times[i * ITERS : i * ITERS + ITERS]))
    return maxs


def plot(x: list, y: list, std: list, param: str):
    plt.figure()
    plt.title("Varying parameter: " + param_name[param])
    plt.plot(x, y, label=param)
    plt.errorbar(x, y, yerr=std, fmt="o", label="Standard Deviation")
    plt.ylabel("Time (s)")
    plt.xlabel(param_name[param] + " (" + param + ")")
    plt.legend(loc="best")


if __name__ == "__main__":
    cli_args = []
    arg_vals = {}
    arg_vals["N"] = [100, 200, 300, 400, 500]
    arg_vals["dt"] = [0.001, 0.002, 0.003, 0.004, 0.005]
    arg_vals["nu"] = [0.001, 0.002, 0.003, 0.004, 0.005]

    if len(sys.argv) == 1:
        cli_args = ["N", "dt", "nu"]
    elif len(sys.argv) != 2:
        print("Insufficient or excessive number of arguements, please provide at most one argument or none.")
        exit(0)
    else:
        match sys.argv[1]:
            case "res":
                cli_args.append("N")
            case "time":
                cli_args.append("dt")
            case "visc":
                cli_args.append("nu")
            case _:
                print("Please provide valid argument! [res/time/visc]")
                exit(0)
        
    avg, std, minimum, maximum = {}, {}, {}, {}
    for param in cli_args:
        times = get_measurements(param, arg_vals[param])
        avg[param] = get_avg(times)
        std[param] = get_std(times)
        minimum[param] = get_min(times)
        maximum[param] = get_max(times)
        plot(arg_vals[param], avg[param], std[param], param)
        
    plt.show()
