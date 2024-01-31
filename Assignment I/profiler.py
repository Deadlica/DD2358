import psutil as psu
import matplotlib.pyplot as plt
from functools import wraps
import threading
from timeit import default_timer as timer
import tkinter as tk
from tkinter import ttk

CORES = psu.cpu_count(logical=True)

def plot_results(cpu_stats, timepoints):
    """
    This functions takes the cpu values of all cores and plots them
    as timeseries figures.
    :param cpu_stats: List of lists where each list are the values of a specific cores at all timepoints
    :param timepoints: All the timepoints that were captured
    :return: None
    """
    plt.figure("CPU Usage Per Core (%)")
    counter = 0
    for i in range(4):
        for j in range(4):
            ax = plt.subplot2grid((4,4), (i,j))
            ax.plot(timepoints, cpu_stats[counter], label=f"Core {counter+1}")
            ax.set_ylim([-1, 101])
            ax.legend()
            if(i == 3):
                plt.xlabel("Time (s)")
            if(j == 0):
                plt.ylabel("CPU usage (%)")
            counter += 1
    plt.show()

def table_results(cpu_stats, timepoints):
    """
    This functions takes the the cpu values of all cores and displays
    them in a table
    :param cpu_stats: List of lists where each list are the values for all cores at a timepoint
    :param timepoints: All the timepoints that were captured
    :return: None
    """
    window = tk.Tk()
    window.title("CPU Core Usage (%)")

    table = ttk.Treeview(window)
    cols = []
    for i in range(CORES):
        cols.append("Core " + str(i + 1))
    table["columns"] = tuple(cols)


    table.column("#0", width=200, minwidth=200)
    table.heading("#0", text="Timepoint")
    for column in table["columns"]:
        table.column(column, width=100, minwidth=100)
        table.heading(column, text=column)

    # Add data to the table
    for i in range(len(cpu_stats)):
        table.insert(parent="", index="end", iid=i, text=timepoints[i], values=tuple(cpu_stats[i]))

    # Pack the table widget
    table.pack(expand=True, fill="both")

    # Run the Tkinter event loop
    window.mainloop()

def monitor_cpu(graph_cpu_stats, table_cpu_stats, timepoints):
    """
    This functions captures CPU usage for all cores during it's monitoring period
    :param graph_cpu_stats: List of lists for the values to be used for plotting
    :param table_cpu_stats: List of lists for the values to be used for table
    :param timepoints: List for the timepoint at each interval
    :return: None
    """
    start = timer()
    while True:
        vals = psu.cpu_percent(interval=0.1, percpu=True)
        table_cpu_stats.append(vals)
        timepoints.append(timer() - start)
        for i in range(len(vals)):
            graph_cpu_stats[i].append(vals[i])
        if not monitoring:
            break

def profile(fn):
    """
    This is a decorator for profiling CPU usage of functions
    :param fn: The function to be profiled
    :return: None
    """
    @wraps(fn)
    def cpu_per_core(*args, **kwargs):
        global monitoring
        cpu_stats = [[] for _ in range(CORES)]
        table_stats = []
        timepoints = []

        monitoring = True
        monitor_thread = threading.Thread(target=monitor_cpu, args=(cpu_stats, table_stats, timepoints,))
        monitor_thread.start()

        fn(*args, **kwargs)

        monitoring = False
        monitor_thread.join()

        table_thread = threading.Thread(target=table_results, args=(table_stats, timepoints,))
        table_thread.start()
        plot_results(cpu_stats, timepoints)
        table_thread.join()
    return cpu_per_core
